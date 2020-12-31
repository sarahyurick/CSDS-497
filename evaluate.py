import argparse
import json
from evaluation_metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("-vector_file", required=True)  # a json
parser.add_argument("-evaluate_file", required=True)  # a txt file
# all of these additional arguments are specific to Sarah's evaluation
parser.add_argument("-metric", required=False)
parser.add_argument("-k", required=False, type=int)
parser.add_argument("-allow_OOV", required=False)

options = parser.parse_args()
vector_file = options.vector_file
evaluate_file = options.evaluate_file
metric = options.metric
k = options.k
allow_OOV = options.allow_OOV

if k is None:
    k = -1
if allow_OOV is None:
    allow_OOV = False
else:
    allow_OOV = False

with open(vector_file) as json_file:
    word_vectors = json.load(json_file)
word_vectors = util.dict_of_list_values_to_arrays(word_vectors)
print("Loaded vector_file")

with open(evaluate_file, 'r') as f:
    test_string = f.read().replace('\n', ' ')
    print("Loaded evaluate_file")

if metric is None:
    bigram_score = calculate_bigram_correlation(test_string, word_vectors, measure="cos")
    print("Bi-gram score using cosine similarity (between -1 and 1):", bigram_score)
    bigram_score = calculate_bigram_correlation(test_string, word_vectors, measure="euclidean")
    print("Bi-gram score using euclidean distance (between -1 and 1):", bigram_score)

    perplexity_score = calculate_perplexity(test_string, word_vectors,
                                            k=k, allow_OOV=allow_OOV, measure="cos")
    print("Perplexity using cosine similarity:", perplexity_score)
    perplexity_score = calculate_perplexity(test_string, word_vectors,
                                            k=k, allow_OOV=allow_OOV, measure="euclidean")
    print("Perplexity using euclidean distance:", perplexity_score)

elif "bigram" in metric:
    bigram_score = calculate_bigram_correlation(test_string, word_vectors, measure="cos")
    print("Bi-gram score using cosine similarity (between -1 and 1):", bigram_score)
    bigram_score = calculate_bigram_correlation(test_string, word_vectors, measure="euclidean")
    print("Bi-gram score using euclidean distance (between -1 and 1):", bigram_score)

elif "perplexity" in metric:
    perplexity_score = calculate_perplexity(test_string, word_vectors,
                                            k=k, allow_OOV=allow_OOV, measure="cos")
    print("Perplexity using cosine similarity:", perplexity_score)
    perplexity_score = calculate_perplexity(test_string, word_vectors,
                                            k=k, allow_OOV=allow_OOV, measure="euclidean")
    print("Perplexity using euclidean distance:", perplexity_score)
