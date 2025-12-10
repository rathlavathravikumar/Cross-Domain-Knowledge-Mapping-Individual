import spacy
 
nlp = spacy.load("en_core_web_sm")
 
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

                relation = token.lemma_  # base form

            if "obj" in token.dep_:

                object_ = token.text

        if subject and relation and object_:

            triples.append((subject, relation, object_))

    return triples
 
# Test it

example = "Albert einstein found the theory of relativity in 1905"

print(extract_relations(example))

 
