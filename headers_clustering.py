from sklearn.cluster import MiniBatchKMeans
import numpy as np
from tqdm import tqdm
CLUSTERS = 100
FILENAME = "vectors/titles_embeddings.txt"
SENT_FILENAME = "vectors/titles.txt"

km = MiniBatchKMeans(n_clusters=CLUSTERS, verbose=1, n_init=500, reassignment_ratio=0.1, random_state=100)

sentence_embeddings = [np.array([float(a) for a in l.strip().split()])
                       for l in tqdm(open(FILENAME, "r", encoding="utf-8"))]
sentence_embeddings = [a / np.linalg.norm(a) for a in sentence_embeddings]
sentences = [l.strip() for l in tqdm(open(SENT_FILENAME, "r", encoding="utf-8"))]

m = np.array(sentence_embeddings)
km.fit(m)
labels = km.labels_
clusters_map = {l: [] for l in labels}

for i, (s, l) in enumerate(zip(sentences, labels)):
    clusters_map[l].append((i, s))

for l in sorted(list(clusters_map.keys())):
    print("cluster %d (%d sentences)" % (l, len(clusters_map[l])))
    for i, header in clusters_map[l][:5]:
        print("    (header #%d) %s " % (i, header))
    print("...")
