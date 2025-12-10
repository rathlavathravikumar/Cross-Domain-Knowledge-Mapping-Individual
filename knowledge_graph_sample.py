import pandas as pd
import networkx as nx
from pyvis.network import Network
import webbrowser
import ast

sample_data = {
    "Triples": [
        str([
            ("Alice", "works_at", "Google"),
            ("Bob", "works_at", "Microsoft")
        ]),
        str([
            ("Charlie", "lives_in", "New York"),
            ("Alice", "lives_in", "California")
        ]),
        str([
            ("Google", "located_in", "USA"),
            ("Microsoft", "located_in", "USA")
        ]),
        str([
            ("Alice", "knows", "Bob"),
            ("Bob", "knows", "Charlie")
        ]),
        str([
            ("Charlie", "studied_at", "MIT"),
            ("Alice", "graduated_from", "Stanford")
        ])
    ]
}

# Convert to DataFrame (simulating CSV read)
data = pd.DataFrame(sample_data)
print("✅ Sample dataset created with", len(data), "rows")


#  Extract triples from dataset

triples_list = []

for cell in data.iloc[:, -1]:  # last column
    try:
        tuples = ast.literal_eval(cell)  # safer than eval
        triples_list.extend(tuples)
    except Exception as e:
        print("Error parsing cell:", e)

print(f"✅ Extracted {len(triples_list)} triples from sample dataset\n")

#  Build Knowledge Graph

G = nx.DiGraph()

for subj, rel, obj in triples_list:
    G.add_node(subj, color="lightblue")
    G.add_node(obj, color="lightgreen")
    G.add_edge(subj, obj, label=rel)

print(f"✅ Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.\n")

# Print first few relationships
for node in list(G.nodes())[:5]:
    if len(G[node]) > 0:
        print(f"Relations for '{node}':")
        for neighbor in G[node]:
            print("→", neighbor, "| Relation:", G[node][neighbor]['label'])
        print()


# Visualize graph with PyVis (HTML)

net = Network(height="750px", width="100%", directed=True, notebook=False)
net.from_nx(G)
net.toggle_physics(True)
net.show_buttons(filter_=['physics'])  # optional: allows tuning in browser
net.write_html("my_knowledge_graph1.html")

print("🎉 Interactive Knowledge Graph saved as my_knowledge_graph1.html")
webbrowser.open("my_knowledge_graph1.html")
