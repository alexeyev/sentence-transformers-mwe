import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import kneighbors_graph

FILENAME = "ethnic_food_top10categories-vectorized.tsv"
K = 2
COSINE_DISTANCE_CUTOFF = 0.15
REVIEWS_GROUPING = False

try:
    import pandas as pd

    df = pd.read_csv(FILENAME, sep="\t").head(5000)
    print("Data read: %s." % FILENAME)
    sentence_embeddings = [np.array([float(num) for num in arr.split()]) for arr in df.embeddings]
    review_ids = list(df.review_id.tolist())
    sentences = list(df.text.tolist())
    categories = list(df.food_categories.tolist())
    print(len(review_ids))
    print(len(sentences))
    del df
except Exception as e:
    print(e)
    quit()
    texts = "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for. They've got about 12 seats outside to go along with the inside.   The smoked meat is up there in quality and taste with Schwartz's and you'll find less tourists at Lester's as well."
    sentences = sent_tokenize(texts, language="english")
    model = SentenceTransformer("./bert-base-nli-mean-tokens")
    sentence_embeddings = model.encode(sentences)

available_colors = ["green", "blue", "yellow", "cyan", "0.75", "0.5", "black", "red", "magenta", "white", "0.25"]
categories_set = list(set(categories))
print("A total of %d categories" % len(categories_set))

cuisine2color = {cui: col for cui, col in zip(categories_set, available_colors)}

print("Graph construction...")
distances = kneighbors_graph(sentence_embeddings, n_neighbors=K * 3, mode="distance",
                             include_self=False, n_jobs=6, metric="cosine")
print("Conversion...")
distances = distances.todense()

if REVIEWS_GROUPING:

    grouped_by_reviews = {r: [] for r in set(review_ids)}

    for idx, sentence in enumerate(sentences):
        r_id = review_ids[idx]
        nn_dist = distances[idx].A1
        similar_sentence_ids = nn_dist.nonzero()[0]
        similar_sentence_rids = [(review_ids[i], nn_dist[i]) for i in similar_sentence_ids if review_ids[i] != r_id]
        grouped_by_reviews[r_id].extend(similar_sentence_rids)

    for rid in grouped_by_reviews:
        closest = sorted(grouped_by_reviews[rid], key=lambda x: x[1], reverse=False)
        unique_closest = [(k, 1 - v) for k, v in closest if v < COSINE_DISTANCE_CUTOFF]
        grouped_by_reviews[rid] = unique_closest[:K]
        print(unique_closest)

    rid2review = {r: "" for r in review_ids}

    for idx, rid in enumerate(review_ids):
        rid2review[rid] += sentences[idx] + " "

    # rid2aggvector = {}
    #
    # for idx, rid in enumerate(review_ids):
    #     rid2review[rid] += sentences[idx] + " "

    print("Construction done. Plotting...")

    import networkx as nx
    import pylab as plt

    G = nx.DiGraph()

    for idx, rid in enumerate(review_ids):
        if G.has_node(rid):
            G.nodes[rid]["cuisine"] = categories[idx]
            G.nodes[rid]["color"] = cuisine2color[categories[idx]]
            G.nodes[rid]["text"] = rid2review[rid]
        else:
            G.add_node(rid, cuisine=categories[idx], text=rid2review[rid], color=cuisine2color[categories[idx]])
            for other_rid, weight in grouped_by_reviews[rid]:
                G.add_edge(rid, other_rid) #, weight=weight)

    print("Preparing layout...")
    # pos = nx.spring_layout(G, iterations=50)
    pos = nx.fruchterman_reingold_layout(G, )

    plt.figure(num=None, figsize=(50, 50))
    plt.axis('off')
    fig = plt.figure(1)

    print("Actually drawing...")
    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=100)

    nodes = list(G.nodes())
    node_colors = [G.nodes[n]["color"] for n in nodes]
    nx.draw_networkx_nodes(G, pos=pos, node_size=500, nodelist=nodes, node_color=node_colors)
    plt.savefig("labels.png", bbox_inches="tight")
    plt.close()
else:

    sentences2knn = {i: [] for i, s in enumerate(sentences)}

    for idx, sentence in enumerate(sentences):
        nn_dist = distances[idx].A1
        similar_sentence_ids = nn_dist.nonzero()[0]
        sentences2knn[idx].extend([(i, nn_dist[i]) for i in similar_sentence_ids])

    print("Construction done. Plotting...")

    import networkx as nx
    import pylab as plt

    G = nx.DiGraph()

    for idx in sentences2knn:
        if G.has_node(idx):
            G.nodes[idx]["cuisine"] = categories[idx]
            G.nodes[idx]["color"] = cuisine2color[categories[idx]]
            G.nodes[idx]["text"] = sentences[idx]
        else:
            G.add_node(idx, cuisine=categories[idx], text=sentences[idx], color=cuisine2color[categories[idx]])
            for other_id, weight in sentences2knn[idx]:
                G.add_edge(idx, other_id)  # , weight=weight)

    print("Preparing layout...")
    # pos = nx.spring_layout(G, iterations=50)
    pos = nx.fruchterman_reingold_layout(G)

    plt.figure(num=None, figsize=(50, 50))
    plt.axis('off')
    fig = plt.figure(1)

    print("Actually drawing...")
    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=100)

    nodes = list(G.nodes())
    node_colors = [G.nodes[n]["color"] for n in nodes]
    nx.draw_networkx_nodes(G, pos=pos, node_size=500, nodelist=nodes, node_color=node_colors)
    plt.savefig("labels-sentences.png", bbox_inches="tight")
    plt.close()