# coding: utf-8
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nltk.download('punkt')


def split(r):
    sentences = [s.strip() for s in sent_tokenize(r)]
    return sentences


# business_id,user_id,review_id,food_categories,text
df = pd.read_csv("/media/data/datasets/yelp/food/ethnic_food_top10categories-shuffled.csv")

new_df = [{}]

for _, row in tqdm(df.iterrows()):
    try:
        sentences = split(row["text"])
        for i, s in enumerate(sentences):
            new_df[-1]["text"] = s
            new_df[-1]["sentence_num"] = i + 1
            new_df[-1]["food_categories"] = row["food_categories"]
            new_df[-1]["review_id"] = row["review_id"]
            new_df[-1]["user_id"] = row["user_id"]
            new_df[-1]["business_id"] = row["business_id"]
    except Exception as e:
        print(e)

new_df = pd.DataFrame(new_df)
new_df.to_csv("/media/data/datasets/yelp/food/ethnic_food_top10categories-splitted.csv")