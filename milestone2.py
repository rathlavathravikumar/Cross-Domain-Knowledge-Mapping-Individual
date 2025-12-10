import os
import pandas as pd
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from pyvis.network import Network
import webbrowser

tqdm.pandas()
nlp = spacy.load("en_core_web_sm")

# Load dataset
data = pd.read_csv("KMapFinal.csv")
print("Dataset loaded successfully!\n")

#Named Entity Recognition 
def extract_entities(text):
    doc = nlp(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]

data["entities"] = data["article"].progress_apply(extract_entities)
print("Named Entity Recognition completed!\n")

#Relation Extraction (RE)

def extract_relations(text):
    doc = nlp(str(text))
    triples = []
    for sent in doc.sents:
        subject, relation, object_ = None, None, None
        for token in sent:
            if "subj" in token.dep_:
                subject = token.text
            if token.pos_ == "VERB":
                relation = token.lemma_
            if "obj" in token.dep_:
                object_ = token.text
        if subject and relation and object_:
            triples.append((subject, relation, object_))
    return triples

data["relations"] = data["article"].progress_apply(lambda x: extract_relations(str(x)))
print("Relation Extraction completed!\n")

# Save to CSV for graph visualization
data.to_csv("relations_output.csv", index=False)
print("Relations saved to relations_output.csv\n")

#Knowledge Graph Visualization
# Load your extracted triples
data_graph = pd.read_csv("relations_output.csv").head(50)

triples_list = []
for cell in data_graph.iloc[:, -1]:  # assuming last column contains triples
    try:
        tuples = eval(cell)
        triples_list.extend(tuples)
    except:
        pass

print(f"Extracted {len(triples_list)} triples from relations_output.csv")

# Build Knowledge Graph
G = nx.DiGraph()
for subj, rel, obj in triples_list:
    G.add_node(subj, color="lightblue")
    G.add_node(obj, color="lightgreen")
    G.add_edge(subj, obj, label=rel)

print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.\n")

# Save and open interactive HTML
net = Network(height="750px", width="100%", directed=True, notebook=False)
net.from_nx(G)
net.toggle_physics(True)
net.write_html("my_knowledge_graph.html")

print("Interactive Knowledge Graph saved as my_knowledge_graph.html")
webbrowser.open("my_knowledge_graph.html")

#Semantic Search
print("\nLoading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = data["article"].dropna().tolist()
embeddings = model.encode(sentences, convert_to_tensor=True)

query = input("\nEnter a search query : ")
query_embedding = model.encode(query, convert_to_tensor=True)

cosine_scores = util.cos_sim(query_embedding, embeddings)
top_results = cosine_scores[0].argsort(descending=True)[:5]

print("\nTop 5 Semantic Matches:")
for idx in top_results:
    print(f" ## {sentences[idx]}  |  Score: {float(cosine_scores[0][idx]):.3f}")

print("\nFull pipeline completed successfully!")
