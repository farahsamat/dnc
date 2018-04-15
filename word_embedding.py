import warnings

warnings.filterwarnings('ignore')
import pickle
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])


def vectorize(texts):
    for doc in nlp.pipe(texts, n_threads=8):
        yield doc.vector

# Load data
print("Loading pickled cleaned text inputs...")
with open('x_raw.pickle', 'rb') as f:
    x = pickle.load(f)

print("Vectorizing...")
doc_vectors = list(vectorize(x))

print("Pickling  word embedding...")
with open('embedding.pickle', 'wb') as f:
    pickle.dump(doc_vectors, f)

print("ALL DONE!")
