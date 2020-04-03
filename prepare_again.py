import pandas as pd

sentences_file = "/media/data/datasets/yelp/food/ethnic_food_top10categories-sentences-vectors.tsv"
sentences_file = pd.read_csv(sentences_file, sep="\t")
#sentences_file.columns = ["review_id", "sub_id", "categories", "text"]

from sentence_transformers import SentenceTransformer, models
import numpy as np
from tqdm import tqdm_notebook

model = SentenceTransformer("bert-base-nli-mean-tokens")

sentences_file["embeddings"] = model.encode(sentences_file.text.str.lower().tolist()[:1000], 
                                            batch_size=512, 
                                            show_progress_bar=True)

sentences_file["embeddings"] = sentences_file.embeddings.map(lambda x: " ".join([str(num) for num in x]))

sentences_file.to_csv("/media/data/datasets/yelp/food/ethnic_food_top10categories-sentences-vectors-prepared.tsv", sep="\t")
