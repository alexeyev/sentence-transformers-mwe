from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

df = pd.read_csv("/media/data/datasets/yelp/food/ethnic_food_top10categories-splitted.csv").head(100000)

print(df.text.tolist()[:10])

model = SentenceTransformer("bert-base-nli-mean-tokens")  
df["embeddings"] = model.encode(df.text.tolist(), batch_size=512, show_progress_bar=True)
df["embeddings"] = df.embeddings.map(lambda x: " ".join([str(f) for f in list(x)]))

df.to_csv("/media/data/datasets/yelp/food/ethnic_food_top10categories-vectorized.tsv", sep="\t")
