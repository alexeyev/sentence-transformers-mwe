import scipy
from sentence_transformers import SentenceTransformer

# choose the one you like best:
# https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
# model = SentenceTransformer('bert-base-nli-mean-tokens')
model = SentenceTransformer("./bert-base-nli-mean-tokens")

sentences = ["This framework generates embeddings for each input sentence",
             "Sentences are passed as a list of string.",
             "The quick brown fox jumps over the lazy dog."]

sentence_embeddings = model.encode(sentences)

queries = sentences
query_embeddings = model.encode(queries)

for query, query_embedding in zip(queries, query_embeddings):
    print("\nQuery:", query)
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine").flatten()
    print(distances)