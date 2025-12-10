import pandas as pd
import spacy
import os
from tqdm import tqdm

tqdm.pandas()
nlp = spacy.load("en_core_web_sm")

data = pd.read_csv("news_sample.csv").head(500)

print("✅ Dataset loaded successfully!")
print("📄 Columns found:", data.columns.tolist(), "\n")
print(data.head(), "\n")

text_column = None
for col in data.columns:
    if 'text' in col.lower() or 'content' in col.lower() or 'article' in col.lower():
        text_column = col
        break

if text_column is None:
    raise ValueError("❌ Couldn't find a text column. Please rename your column to 'text' or 'content'.")

print(f"🧾 Using column: {text_column}\n")

def extract_entities(text):
    doc = nlp(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]

data["entities"] = data[text_column].progress_apply(extract_entities)

print("===== Sample Extracted Entities =====")
print(data[[text_column, "entities"]].head(), "\n")

os.makedirs("data/processed", exist_ok=True)
data.to_csv("data/processed/entities_extracted.csv", index=False)
print("✅ Entities saved to data/processed/entities_extracted.csv")
