import spacy

nlp = spacy.load("en_core_web_sm")

text = "Albert Einstein developed the theory of relativity in 1905. Elon Musk founded SpaceX in 2002."

doc = nlp(text)

print("Entities and their labels:")
for ent in doc.ents:
    print(f"{ent.text:<25} | {ent.label_}")

print("\nLabel meanings:")
print(spacy.explain("ORG"))
