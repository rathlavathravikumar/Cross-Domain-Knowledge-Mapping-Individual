import spacy
import pandas as pd

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to extract subject-relation-object triples from a sentence
def extract_relations(text):
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        subject = None
        relation = None
        object_ = None
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

# Load your dataset 
df = pd.read_csv("news_sample.csv").head(500)  
text_column = "article"                  

# Extract relations for each row
df["relations"] = df[text_column].apply(lambda x: extract_relations(str(x)))

# Save output to CSV
df.to_csv("relations_output.csv", index=False)

print("✅ Relation extraction completed! Results saved to 'relations_output.csv'")
print(df.head())
