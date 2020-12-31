import argparse
import json
import util
import evaluation_metrics
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-preference", required=True)  # tv show name, e.g. friends
parser.add_argument("-num_recommendations", required=False, type=int)  # how many titles to recommend

options = parser.parse_args()
preference = options.preference
num_recommendations = options.num_recommendations

preference = preference.lower()
if num_recommendations is None:
    num_recommendations = 10

# get list of all titles to choose from
titles = pd.read_csv("netflix_titles.csv")
titles = titles['title'].str.replace(' ', '')
title_corpus = ' '.join(titles.tolist()).lower()
titles = title_corpus.split()

with open("vectors\\netflix_vectors.json") as json_file:
    word_vectors = json.load(json_file)
word_vectors = util.dict_of_list_values_to_arrays(word_vectors)
print("Loaded vector_file")

# calculate distance between preference and all words in our vocab
most_similar_words = evaluation_metrics.find_k_most_similar(preference, word_vectors, -1, measure="euclidean")
recommended_titles = []
count = 0
for word_vector in most_similar_words:
    title = word_vector[0]
    if title in titles:  # figure out which words are actually titles
        recommended_titles.append(title)
        count = count + 1
    if count >= num_recommendations:
        break

print("Your recommended titles:", recommended_titles)
