import spacy

nlp = spacy.load('en_core_web_md')

def generate_embedding(text):
    doc = nlp(text)
    return doc.vector.tolist()
