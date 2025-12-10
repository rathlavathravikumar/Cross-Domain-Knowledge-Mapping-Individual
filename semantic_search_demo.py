from sentence_transformers import SentenceTransformer, util
 
# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')
 
sentences = [
    "Artificial Intelligence connects multiple disciplines.",
    "Machine Learning is a branch of AI.",
    "The theory of relativity revolutionized physics.",
    "Elon Musk founded SpaceX to explore space."
]
 
# Encode sentences
embeddings = model.encode(sentences, convert_to_tensor=True)
 
# Example query
query = "What is related to AI?"
query_embedding = model.encode(query, convert_to_tensor=True)
 
# Compute cosine similarity
cosine_scores = util.cos_sim(query_embedding, embeddings)
 
# Rank results
for i, score in enumerate(cosine_scores[0]):
    print(f"{sentences[i]} --> Score: {score:.3f}")
