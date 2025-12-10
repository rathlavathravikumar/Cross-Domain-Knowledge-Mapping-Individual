from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences dataset (modern technology and discoveries)
sentences = [
    "ChatGPT was developed by OpenAI to understand and generate human-like text.",
    "Blockchain technology enables secure and transparent digital transactions.",
    "CRISPR allows scientists to edit genes with high precision.",
    "The iPhone revolutionized the smartphone industry after its release in 2007.",
    "The Mars Rover Perseverance is exploring the surface of Mars for signs of life.",
    "5G networks offer faster internet speeds and improved connectivity.",
    "Virtual Reality immerses users in a simulated 3D environment.",
    "Renewable energy sources like solar and wind reduce carbon emissions.",
    "Quantum computing uses quantum bits to perform complex calculations faster.",
    "Neural networks are inspired by the human brain and power deep learning.",
    "Tesla produces electric vehicles to promote sustainable transportation.",
    "The Internet of Things connects everyday devices to the internet.",
    "Cloud computing allows data storage and access from anywhere in the world.",
    "Artificial Intelligence is transforming industries through automation."
]

# Encode sentences
embeddings = model.encode(sentences, convert_to_tensor=True)

# Example query
query = "What uses quantum bits?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarity
cosine_scores = util.cos_sim(query_embedding, embeddings)

# Display ranked results
print(f"\nQuery: {query}\n")
for i, score in enumerate(cosine_scores[0]):
    print(f"{sentences[i]} --> Score: {score:.3f}")
