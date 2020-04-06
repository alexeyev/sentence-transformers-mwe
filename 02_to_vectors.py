from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

df = pd.read_csv("/media/data/datasets/yelp/food/ethnic_food_top10categories-sentences-vectors.tsv", sep="\t")#.head(5000)

model = SentenceTransformer("bert-base-nli-mean-tokens")  
df["embeddings"] = model.encode(df.text.str.lower().tolist(), batch_size=512, show_progress_bar=True)
df["embeddings"] = df.embeddings.map(lambda x: " ".join([str(f) for f in list(x)]))

df.to_csv("/media/data/datasets/yelp/food/that_thing.tsv", sep="\t")
