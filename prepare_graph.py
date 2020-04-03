from sklearn.neighbors import kneighbors_graph


from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize

FILENAME = "somefile"

try:
    import pandas as pd
    df = pd.read_csv(FILENAME, sep="\t")
    sentence_embeddings = [np.array([float(num) for num in arr.split()]) for arr in df.embeddings]
    review_ids = df.review_id.tolist()
    sentences = df.text.str.tolist()
    del df
except Exception as e:
    texts = "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for. They've got about 12 seats outside to go along with the inside.   The smoked meat is up there in quality and taste with Schwartz's and you'll find less tourists at Lester's as well."
    sentences = sent_tokenize(texts, language="english")
    model = SentenceTransformer("./bert-base-nli-mean-tokens")
    sentence_embeddings = model.encode(sentences)

distances = kneighbors_graph(sentence_embeddings,
                             n_neighbors=2,
                             mode="connectivity",
                             include_self=False,
                             n_jobs=4,
                             metric="cosine")
distances = distances.todense()

for idx, sentence in enumerate(sentences):
    print(idx, sentence)
    print(">    " + "\n>    ".join([sentences[id] for id in distances[idx].nonzero()[1]]))
    print()