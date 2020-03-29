# coding: utf-8
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

nltk.download('punkt')
model = SentenceTransformer("./bert-base-nli-mean-tokens")
FILEPATH = "food_reviews.txt"

with open(FILEPATH, "r+") as rf:
    with open(FILEPATH + ".reviews", "w+") as wtext:
        with open(FILEPATH + ".vectors", "w+") as wvectors:

            for line in tqdm(rf, "lines of " + FILEPATH):
                sentences = sent_tokenize(line, language="english")
                sentence_embeddings = model.encode(sentences)

                for sentence, emb in zip(sentences, sentence_embeddings):
                    wtext.write(sentence.strip() + "\n")
                    wvectors.write(" ".join([str(v) for v in list(emb)]) + "\n")

                wtext.write("\n")
                wvectors.write("\n")