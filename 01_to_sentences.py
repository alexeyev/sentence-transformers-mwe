# coding: utf-8
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import csv

nltk.download('punkt')


def split(r):
    sentences = [s.strip() for s in sent_tokenize(r)]
    return sentences


# business_id,user_id,review_id,food_categories,text
df = pd.read_csv("/media/data/datasets/yelp/food/ethnic_food_top10categories-shuffled.csv")

new_df = [{}]

with open("/media/data/datasets/yelp/food/ethnic_food_top10categories-splitted.csv", "w+") as wf:

    csvw = csv.writer(wf)
    csvw.writerow("business_id,user_id,review_id,food_categories,text,sentence_num".split(","))

    for _, row in tqdm(df.iterrows()):
        try:
            sentences = split(row["text"])
            for i, s in enumerate(sentences):
                csvw.writerow([row["business_id"], row["user_id"], row["review_id"],
                               row["food_categories"], s, i + 1])
        except Exception as e:
            print(e)
