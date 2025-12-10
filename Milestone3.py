# mental_health_dashboard_final.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import tempfile
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import speech_recognition as sr
import spacy
from datetime import datetime
import os

st.set_page_config(page_title="Mental Health Detection", layout="wide")

# ------------------ CSS ------------------
st.markdown(
    """
    <style>
    input, textarea, select { color: black !important; background-color: white !important; }
    ::placeholder { color: #6c757d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Session state initialization ------------------
if "users" not in st.session_state:
    st.session_state.users = {}
if "auth" not in st.session_state:
    st.session_state.auth = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "model_classes" not in st.session_state:
    st.session_state.model_classes = []

if "kg" not in st.session_state:
    st.session_state.kg = nx.DiGraph()
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# pipeline status (simple simulated statuses stored in session)
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {
        "ETL Pipeline": "Idle",
        "Graph Updater": "Idle"
    }

# ------------------ Small disorder KB ------------------
DISORDER_KB = {
    "depression": ["sadness", "loss of interest", "fatigue", "sleep issues", "low appetite", "hopelessness"],
    "anxiety": ["worry", "restlessness", "panic", "sleep issues", "irritability"],
    "bipolar": ["mania", "depression", "mood swings", "impulsivity"],
    "adhd": ["inattention", "hyperactivity", "impulsivity"],
    "ocd": ["obsessions", "compulsions", "anxiety"]
}

# ------------------ spaCy model ------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# ------------------ Utilities ------------------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def add_dataset_to_session(df, name="dataset"):
    st.session_state.df = df
    st.session_state.df_name = name

def build_graph_from_dataset(sample_n=200):
    if st.session_state.df is None:
        return
    G = st.session_state.kg
    df_sample = st.session_state.df.sample(min(len(st.session_state.df), sample_n), random_state=1)
    for _, row in df_sample.iterrows():
        t = row["text"].strip()[:120]
        l = str(row["label"])
        if t and not G.has_node(t):
            G.add_node(t, type="text")
        if not G.has_node(l):
            G.add_node(l, type="label")
        if t:
            G.add_edge(t, l)
    st.session_state.kg = G

def show_pyvis(G, height=650):
    net = Network(height=f"{height}px", width="100%", directed=True)
    net.from_nx(G)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    return tmp.name

# ------------------ Model Training ------------------
def train_and_store(df):
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vect = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    st.session_state.vectorizer = vect
    st.session_state.model = clf
    st.session_state.trained = True
    st.session_state.model_classes = list(clf.classes_)

    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return acc, fig

# ------------------ Prediction + Graph Update ------------------
def predict_and_update_graph(text):
    if not st.session_state.trained:
        return None

    vec = st.session_state.vectorizer.transform([text])
    pred = st.session_state.model.predict(vec)[0]

    G = st.session_state.kg
    text_node = text.strip()[:200]

    if not G.has_node(text_node):
        G.add_node(text_node, type="text")

    if not G.has_node(pred):
        G.add_node(pred, type="label")

    G.add_edge(text_node, pred)

    if pred.lower() in DISORDER_KB:
        for rel in DISORDER_KB[pred.lower()]:
            if not G.has_node(rel):
                G.add_node(rel, type="concept")
            G.add_edge(pred, rel, relation="related")

    st.session_state.kg = G

    st.session_state.pred_history.append({
        "user": st.session_state.current_user,
        "text": text,
        "predicted": pred,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    return pred

# ------------------ Triplet Extraction ------------------
def extract_triplets(text):
    doc = nlp(text)
    triplets = []
    for sent in doc.sents:
        subj = obj = verb = ""
        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            if "obj" in token.dep_:
                obj = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_
        if subj and verb and obj:
            triplets.append((subj, verb, obj))
    return triplets
# ------------------ Signup UI ------------------
def signup_ui():
    st.header("Signup")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")

    if st.button("Create Account", key="signup_create_btn"):
        if not username or not password:
            st.error("Enter username and password.")
            return
        if username in st.session_state.users:
            st.error("Username exists.")
            return
        st.session_state.users[username] = password
        st.session_state.auth = True
        st.session_state.current_user = username
        st.success("Account created. Logged in.")

# ------------------ Login UI ------------------
def login_ui():
    st.header("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_btn"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.auth = True
            st.session_state.current_user = username
            st.success("Welcome!")
        else:
            st.error("Invalid credentials.")

# ------------------ Admin tools (helpers) ------------------
def merge_nodes_in_graph(G, target, source):
    """
    Merge source node into target node:
    - Redirect all edges (incoming/outgoing) from source to target (avoid self-loops)
    - Remove source node
    """
    if source not in G or target not in G or source == target:
        return False, "Invalid nodes."

    # copy list because we'll modify graph
    for pred in list(G.predecessors(source)):
        if pred != target:
            G.add_edge(pred, target, **G.get_edge_data(pred, source) or {})
    for succ in list(G.successors(source)):
        if succ != target:
            G.add_edge(target, succ, **G.get_edge_data(source, succ) or {})

    # if source had node attributes merge them into target (non-destructive)
    if G.nodes[source]:
        for k, v in G.nodes[source].items():
            if k not in G.nodes[target]:
                G.nodes[target][k] = v

    G.remove_node(source)
    return True, f"Merged '{source}' into '{target}'."

def list_edges_with_attrs(G):
    rows = []
    for u, v, attrs in G.edges(data=True):
        rows.append({
            "source": u,
            "target": v,
            "relation": attrs.get("relation", ""),
            "confirmed": attrs.get("confirmed", False)
        })
    return pd.DataFrame(rows)

# ------------------ Dashboard ------------------
def dashboard_ui():
    st.header(f"Mental Health Detection Dashboard")

    # ---------------- Dataset Upload ----------------
    st.subheader("📂 Upload Dataset (CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df = df.iloc[:, :2]
            df.columns = ["text", "label"]
            df["text"] = df["text"].astype(str).apply(clean_text)
            df = df.dropna()
            add_dataset_to_session(df, name=uploaded.name)
            build_graph_from_dataset()
            st.success(f"Dataset uploaded successfully — {len(df)} rows")
        except Exception as e:
            st.error(f"Invalid CSV format: {e}")

    st.markdown("---")

    tabs = st.tabs([
        "Dataset",
        "Triplet Extraction",
        "Train & Detect",
        "Charts",
        "Graph",
        "Semantic Search",
        "Admin",
        "Feedback"
    ])

    # ---------------- Dataset Tab ----------------
    with tabs[0]:
        if st.session_state.df is None:
            st.info("Upload a dataset first.")
        else:
            st.write("Dataset name:", st.session_state.df_name)
            rows_to_show = st.slider("Preview rows", 5, 200, 20, key="dataset_preview_rows")
            st.dataframe(st.session_state.df.head(rows_to_show))

    # ---------------- Triplet Extraction ----------------
    with tabs[1]:
        st.subheader("Triplet Extraction")
        txt = st.text_area("Enter text for triplet extraction", key="triplet_text_input", height=140)
        if st.button("Extract Triplets", key="extract_triplets_btn"):
            if txt.strip():
                result = extract_triplets(txt)
                if result:
                    st.dataframe(pd.DataFrame(result, columns=["subject", "relation", "object"]))
                else:
                    st.info("No triplets found.")

    # ---------------- Train & Detect ----------------
    with tabs[2]:
        st.subheader("Train Model")
        if st.button("Train", key="train_model_btn"):
            if st.session_state.df is None:
                st.error("Upload dataset first.")
            else:
                try:
                    acc, fig = train_and_store(st.session_state.df)
                    st.success(f"Accuracy: {acc:.3f}")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Training failed: {e}")

        st.subheader("Predict")
        text = st.text_area("Enter text to predict", key="dashboard_text_input", height=140)
        if st.button("Predict Disorder", key="predict_disorder_btn"):
            if not st.session_state.trained:
                st.error("Train model first.")
            else:
                if not text.strip():
                    st.warning("Enter text to predict.")
                else:
                    try:
                        pred = predict_and_update_graph(text)
                        if pred:
                            st.success(f"Prediction: {pred}")
                        else:
                            st.error("Prediction failed.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

        st.markdown("**Prediction History (latest first)**")
        if st.session_state.pred_history:
            st.dataframe(pd.DataFrame(list(reversed(st.session_state.pred_history))).head(50))
        else:
            st.info("No prediction history yet.")

    # ---------------- Charts ----------------
    with tabs[3]:
        if st.session_state.df is None:
            st.info("Upload dataset first.")
        else:
            counts = st.session_state.df["label"].value_counts()
            st.write("Category distribution")
            st.bar_chart(counts)
            # pie chart
            figp, axp = plt.subplots()
            axp.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            axp.axis("equal")
            st.pyplot(figp)

    # ---------------- Graph ----------------
    with tabs[4]:
        G = st.session_state.kg
        st.write(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

        if st.button("Show Full Graph", key="show_full_graph_btn"):
            if len(G.nodes()) == 0:
                st.info("Graph empty.")
            else:
                html = show_pyvis(G)
                st.components.v1.html(open(html, "r", encoding="utf-8").read(), height=650)

    # ---------------- Semantic Search ----------------
    with tabs[5]:
        st.subheader("🔎 Semantic Search & Subgraph generation")

        method = st.radio("Search method", ["Keyword (match node text)", "Semantic (model similarity)"], index=0, key="semantic_method_radio")

        query = st.text_input("Search text / query", key="semantic_search_input")
        topk = st.slider("Top K (for semantic)", 1, 10, 3, key="semantic_topk_slider")
        expand_depth = st.slider("Subgraph neighbor expansion depth", 0, 2, 1, key="semantic_expand_depth")

        if st.button("Search", key="semantic_search_btn"):
            if method == "Semantic (model similarity)" and not st.session_state.trained:
                st.error("Train model first for semantic similarity.")
            elif not query.strip():
                st.warning("Enter query text.")
            else:
                matched_nodes = []
                G = st.session_state.kg
                if method == "Keyword (match node text)":
                    matched_nodes = [n for n in G.nodes() if query.lower() in str(n).lower()]
                    if len(matched_nodes) == 0 and st.session_state.df is not None:
                        # fallback: search dataset texts
                        matched_rows = st.session_state.df[st.session_state.df["text"].str.contains(query, case=False, na=False)]
                        matched_nodes = list(matched_rows["text"].head(topk).values)
                else:
                    # semantic: use vectorizer to find top-k docs, then use their text nodes
                    vecq = st.session_state.vectorizer.transform([query])
                    docs = st.session_state.vectorizer.transform(st.session_state.df["text"])
                    sims = (vecq @ docs.T).toarray().flatten()
                    idxs = sims.argsort()[-topk:][::-1]
                    matched_nodes = list(st.session_state.df.iloc[idxs]["text"].values)

                if not matched_nodes:
                    st.info("No matches found.")
                else:
                    st.success(f"Found {len(matched_nodes)} matching nodes.")
                    # build subgraph by adding neighbors up to expand_depth
                    nodeset = set()
                    for n in matched_nodes:
                        if n in G:
                            nodeset.add(n)
                            if expand_depth >= 1:
                                nodeset.update(G.predecessors(n))
                                nodeset.update(G.successors(n))
                            if expand_depth >= 2:
                                # neighbors of neighbors
                                for nb in list(G.predecessors(n)) + list(G.successors(n)):
                                    nodeset.update(G.predecessors(nb))
                                    nodeset.update(G.successors(nb))
                        else:
                            # if node not in graph but it's raw text from dataset, add it (and its label)
                            nodeset.add(n)
                            # include label if exists in df
                            if st.session_state.df is not None:
                                row = st.session_state.df[st.session_state.df["text"] == n]
                                if not row.empty:
                                    lbl = str(row.iloc[0]["label"])
                                    nodeset.add(lbl)
                    sub = G.subgraph([x for x in nodeset if x in G.nodes()] + [x for x in nodeset if x not in G.nodes()]).copy()
                    # if subgraph is empty but nodeset contains raw texts not in G, create small graph
                    if sub.number_of_nodes() == 0:
                        sub = nx.DiGraph()
                        for n in nodeset:
                            sub.add_node(n, type="text" if n in list(st.session_state.df["text"].values) else "label")
                    html = show_pyvis(sub, height=600)
                    st.components.v1.html(open(html, "r", encoding="utf-8").read(), height=600)
                    st.write(f"Subgraph nodes: {len(sub.nodes())}, edges: {len(sub.edges())}")

        # Option to directly generate subgraph from the query (single button)
        if st.button("Generate Subgraph from Query", key="generate_subgraph_btn"):
            # reuse search logic above
            if not query.strip():
                st.warning("Enter query text.")
            else:
                st.session_state["__semantic_generate_query"] = query
                st.experimental_rerun()

        # If the quick-generate key exists, handle it (this triggers after rerun)
        if "__semantic_generate_query" in st.session_state:
            q = st.session_state.pop("__semantic_generate_query")
            # perform same steps as above but with defaults
            G = st.session_state.kg
            matched_nodes = [n for n in G.nodes() if q.lower() in str(n).lower()]
            if not matched_nodes and st.session_state.trained and st.session_state.df is not None:
                vecq = st.session_state.vectorizer.transform([q])
                docs = st.session_state.vectorizer.transform(st.session_state.df["text"])
                sims = (vecq @ docs.T).toarray().flatten()
                idxs = sims.argsort()[-3:][::-1]
                matched_nodes = list(st.session_state.df.iloc[idxs]["text"].values)
            if not matched_nodes:
                st.info("No matches found for subgraph generation.")
            else:
                nodeset = set()
                for n in matched_nodes:
                    if n in G:
                        nodeset.add(n)
                        nodeset.update(G.predecessors(n))
                        nodeset.update(G.successors(n))
                sub = G.subgraph([x for x in nodeset if x in G.nodes()]).copy()
                if sub.number_of_nodes() == 0:
                    st.info("No nodes in graph for the given query.")
                else:
                    html = show_pyvis(sub, height=600)
                    st.components.v1.html(open(html, "r", encoding="utf-8").read(), height=600)
                    st.success(f"Generated subgraph with {len(sub.nodes())} nodes")

    # ---------------- Admin ----------------
    with tabs[6]:
        st.subheader("Admin Dashboard — Monitor pipelines, datasets & refine graph")

        admin_tabs = st.tabs(["📊 Pipelines", "📁 Dataset Viewer", "🔧 Graph Refinement", "🔎 Edge List & Confirm"])

        # ---------- Pipelines ----------
        with admin_tabs[0]:
            st.write("### Processing Pipelines")
            for pname, status in st.session_state.pipeline_status.items():
                st.info(f"**{pname}** → Status: **{status}**")

            colp1, colp2, colp3 = st.columns([1,1,1])
            if colp1.button("Start ETL Pipeline", key="admin_start_etl_btn"):
                st.session_state.pipeline_status["ETL Pipeline"] = "Running"
                st.success("ETL Pipeline marked as Running (simulated).")
            if colp2.button("Stop ETL Pipeline", key="admin_stop_etl_btn"):
                st.session_state.pipeline_status["ETL Pipeline"] = "Stopped"
                st.warning("ETL Pipeline marked as Stopped (simulated).")
            if colp3.button("Refresh Pipeline Status", key="admin_refresh_pipelines_btn"):
                st.session_state.pipeline_status["Graph Updater"] = "Idle"
                st.success("Pipeline statuses refreshed (simulated).")

        # ---------- Dataset Viewer ----------
        with admin_tabs[1]:
            st.write("### Dataset Viewer")
            if st.session_state.df is None:
                st.info("No dataset uploaded.")
            else:
                st.write(f"*Dataset:* {st.session_state.df_name} — Rows: {len(st.session_state.df)}")
                n_preview = st.number_input("Rows to preview", min_value=1, max_value=min(1000, len(st.session_state.df)), value=20, key="admin_dataset_preview_rows")
                st.dataframe(st.session_state.df.head(n_preview))

                if st.button("Download dataset CSV", key="admin_download_dataset_btn"):
                    csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
                    st.download_button("Click to download", data=csv_bytes, file_name=f"{st.session_state.df_name}", mime="text/csv", key="admin_dl_btn_inner")

        # ---------- Graph Refinement ----------
        with admin_tabs[2]:
            st.write("### Graph Refinement Tools")
            G = st.session_state.kg
            if len(G.nodes()) == 0:
                st.info("Graph empty. Run dataset build or predictions to populate it.")
            else:
                all_nodes = list(G.nodes())
                colm1, colm2 = st.columns(2)
                with colm1:
                    node_source = st.selectbox("Node to merge (source)", all_nodes, key="merge_source_select")
                with colm2:
                    node_target = st.selectbox("Target node (merge into)", all_nodes, key="merge_target_select")

                if st.button("Merge Selected Nodes", key="admin_merge_nodes_btn"):
                    ok, msg = merge_nodes_in_graph(G, node_target, node_source)
                    if ok:
                        st.success(msg)
                        st.session_state.kg = G
                    else:
                        st.error(msg)

                st.markdown("---")
                st.write("Edit / Rename Node")
                edit_node = st.selectbox("Select node to rename", all_nodes, key="admin_edit_node_select")
                new_label = st.text_input("New label", key="admin_new_label_input")
                if st.button("Rename Node", key="admin_rename_node_btn"):
                    if not new_label.strip():
                        st.warning("Enter a new label.")
                    else:
                        try:
                            nx.relabel_nodes(G, {edit_node: new_label}, copy=False)
                            st.success(f"Renamed '{edit_node}' to '{new_label}'.")
                            st.session_state.kg = G
                        except Exception as e:
                            st.error(f"Rename failed: {e}")

                st.markdown("---")
                st.write("Delete Node")
                del_node = st.selectbox("Select node to delete", all_nodes, key="admin_delete_node_select")
                if st.button("Delete Node", key="admin_delete_node_btn"):
                    try:
                        G.remove_node(del_node)
                        st.success(f"Deleted node '{del_node}'.")
                        st.session_state.kg = G
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

                st.markdown("---")
                if st.button("Show small graph preview", key="admin_graph_preview_btn"):
                    html = show_pyvis(G, height=500)
                    st.components.v1.html(open(html, "r", encoding="utf-8").read(), height=500)

        # ---------- Edge list & confirm ----------
        with admin_tabs[3]:
            st.write("### Edge list & Confirm relations")
            G = st.session_state.kg
            if len(G.edges()) == 0:
                st.info("No edges in graph.")
            else:
                edges_df = list_edges_with_attrs(G)
                st.dataframe(edges_df)

                # select an edge
                edge_choices = [f"{r['source']} -> {r['target']} (relation={r['relation']}, confirmed={r['confirmed']})" for _, r in edges_df.iterrows()]
                sel_edge = st.selectbox("Select edge to confirm/edit", [""] + edge_choices, key="admin_select_edge")
                if sel_edge:
                    # parse selection
                    left = sel_edge.split(" -> ")[0]
                    right = sel_edge.split(" -> ")[1].split(" (")[0]
                    cur_relation = G.get_edge_data(left, right).get("relation", "")
                    cur_confirmed = G.get_edge_data(left, right).get("confirmed", False)
                    st.write(f"Current relation: '{cur_relation}', confirmed: {cur_confirmed}")
                    new_relation = st.text_input("Edit relation text", value=cur_relation, key="admin_edit_relation_input")
                    if st.button("Confirm / Update relation", key="admin_confirm_relation_btn"):
                        G[left][right]["relation"] = new_relation
                        G[left][right]["confirmed"] = True
                        st.success(f"Edge {left} -> {right} updated and confirmed.")
                        st.session_state.kg = G
           
          

    # ---------------- Feedback ----------------
    with tabs[7]:
        fb = st.text_area("Your feedback", key="feedback_box", height=140)
        if st.button("Submit Feedback", key="feedback_submit_btn"):
            if fb.strip():
                st.session_state.feedback.append({
                    "user": st.session_state.current_user,
                    "text": fb,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Feedback submitted.")

# ------------------ Main ------------------
def main():
    st.sidebar.title("Account")

    if not st.session_state.auth:
        choice = st.sidebar.radio("Select", ["Signup", "Login"], key="sidebar_auth_choice")
        if choice == "Signup":
            signup_ui()
        else:
            login_ui()
    else:
        dashboard_ui()

if __name__ == "__main__":
    main()
