import networkx as nx
import numpy as np
from networkx import write_gml
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import kneighbors_graph

FILENAME = "vectors/titles_embeddings.txt"
SENT_FILENAME = "vectors/titles.txt"
K = 1
COSINE_DISTANCE_CUTOFF = 0.9
SAMPLE = 60000

try:
    sentence_embeddings = [np.array([float(a) for a in l.strip().split()])
                           for l in open(FILENAME, "r", encoding="utf-8").readlines()][:SAMPLE]
    sentences = [l.strip() for l in open(SENT_FILENAME, "r", encoding="utf-8").readlines()][:SAMPLE]
except Exception as e:
    print(e)
    quit()
    texts = "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for. They've got about 12 seats outside to go along with the inside.   The smoked meat is up there in quality and taste with Schwartz's and you'll find less tourists at Lester's as well."
    sentences = sent_tokenize(texts, language="english")
    model = SentenceTransformer("./bert-base-nli-mean-tokens")
    sentence_embeddings = model.encode(sentences)

print("Graph construction...")
distances = kneighbors_graph(sentence_embeddings, n_neighbors=K * 3, mode="distance",
                             include_self=False, n_jobs=6, metric="cosine")
print("Conversion...")
distances = distances.todense()

sentences2knn = {i: [] for i, s in enumerate(sentences)}

for idx, sentence in enumerate(sentences):
    nn_dist = distances[idx].A1
    similar_sentence_ids = nn_dist.nonzero()[0]
    sentences2knn[idx].extend(
        [(i, 1 - nn_dist[i]) for i in similar_sentence_ids if nn_dist[i] < COSINE_DISTANCE_CUTOFF])
    print(sentences2knn[idx])
    print(sentences[idx])
    for i in similar_sentence_ids:
        print(i, "|", nn_dist[i], sentences[i])
    print()

print("Construction done. Building graph for Gephi...")

G = nx.DiGraph()

for idx in sentences2knn:
    if G.has_node(idx):
        G.nodes[idx]["text"] = sentences[idx]
    else:
        G.add_node(idx, text=sentences[idx])
        for other_id, weight in sentences2knn[idx]:
            G.add_edge(idx, other_id, weight=weight)

write_gml(G, "titles-graph-sentences-%d.gml" % SAMPLE, stringizer=str)
