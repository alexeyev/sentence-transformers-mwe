# coding: utf-8
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

if len(sys.argv) < 2:
    print(sys.argv)
    print("Add an argument please")
    quit()

FILEPATH = sys.argv[-1]

nltk.download('punkt')
model = SentenceTransformer("./bert-base-nli-mean-tokens")

with open(FILEPATH, "r+", errors='ignore') as rf:
    with open(FILEPATH + ".reviews", "w+") as wtext:
        with open(FILEPATH + ".vectors", "w+") as wvectors:

            for line in tqdm(rf, "lines of " + FILEPATH):
                try:
                    sentences = sent_tokenize(line, language="english")
                    sentence_embeddings = model.encode(sentences)

                    for sentence, emb in zip(sentences, sentence_embeddings):
                        wtext.write(sentence.strip() + "\n")
                        wvectors.write(" ".join([str(v) for v in list(emb)]) + "\n")

                    wtext.write("\n")
                    wvectors.write("\n")
                except Exception as e:
                    print("Error:", e)