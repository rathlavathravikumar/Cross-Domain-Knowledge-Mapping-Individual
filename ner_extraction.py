# src/ner_extraction.py

import pandas as pd
import spacy

def load_dataset(file_path):
    """Load CSV dataset"""
    df = pd.read_csv(file_path)
    return df

def extract_entities(df, text_column='article'):
    """Run NER and extract PERSON, ORG, DATE entities"""
    nlp = spacy.load("en_core_web_sm")  # Load spaCy English model
    
    persons = set()
    orgs = set()
    dates = set()

    for doc in nlp.pipe(df[text_column].astype(str), batch_size=50):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.add(ent.text)
            elif ent.label_ == "ORG":
                orgs.add(ent.text)
            elif ent.label_ == "DATE":
                dates.add(ent.text)

    return persons, orgs, dates

def main():
    # Load dataset
    df = load_dataset("data/processed/entities_extracted.csv")  # Adjust path if needed

    # Extract entities
    persons, orgs, dates = extract_entities(df)

    # Print counts
    print(f"Unique PERSON entities: {len(persons)}")
    print(f"Unique ORG entities: {len(orgs)}")
    print(f"Unique DATE entities: {len(dates)}")

if __name__ == "__main__":
    main()
