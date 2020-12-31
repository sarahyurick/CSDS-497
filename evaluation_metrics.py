import util
from scipy.spatial import distance
import tokenizer
import numpy as np
import nltk

unknown_token = "UNK"


def calculate_manhattan_distance(a, b):
    return distance.minkowski(a, b, 1)


def calculate_euclidean_distance(a, b):
    return distance.minkowski(a, b, 2)


def calculate_supremum_distance(a, b):
    return distance.minkowski(a, b, float('inf'))


def calculate_cosine_similarity(a, b):
    return distance.cosine(a, b)


def find_k_most_similar(word, dictionary, k, measure="cos"):
    """
    :param word: the word given, for which we are finding the k most similar words
    :param dictionary: dictionary of form {word: vector}
    :param k: how many similar words to return
    :param measure: "cos" for cosine similarity, "euclidean" for euclidean distance
    :return: list of length k with elements [most_similar_word, similarity], [2nd_most_similar_word, similarity], ...
    """
    word_vector = dictionary[word]
    closest_words = []
    for w in dictionary:
        if w != word:
            try:
                w_vector = dictionary[w]
            except KeyError:
                w_vector = dictionary[unknown_token]
            if measure == "cos":
                sim = calculate_cosine_similarity(word_vector, w_vector)
            elif measure == "euclidean":
                sim = -calculate_euclidean_distance(word_vector, w_vector)
            else:
                raise RuntimeError("Must use cos or euclidean distance metric")
            closest_words.append([w, sim])
    closest_words.sort(key=lambda x: x[1], reverse=True)
    return closest_words[:k]


def calculate_perplexity(evaluation_string, dictionary, k=100, allow_OOV=False, max_length=1000, measure="cos"):
    """
    :param evaluation_string: string for which we are calculating the perplexity of our model
    :param dictionary: our model, dict of form {word: vector}
    :param k: how many of the top similar words to consider. set k=-1 to consider the entire vocabulary
    :param allow_OOV: whether OOV tokens affect perplexity score
    :param max_length: for computational efficiency, only use the first max_length words to calculate perplexity
    :param measure: "cos" for cosine similarity and "euclidean" for euclidean distance
    :return: perplexity value
    """
    counter = 0
    vocab = dictionary.keys()
    if k == -1:
        k = len(vocab)

    evaluation_string = tokenizer.basic(evaluation_string)
    if max_length == -1:
        max_length = len(evaluation_string)
    prev_word = evaluation_string[0]

    probabilities = []

    for word_to_predict in evaluation_string[1:]:
        if (prev_word in vocab and word_to_predict in vocab) or allow_OOV:
            if prev_word not in vocab:
                prev_word = unknown_token
            if word_to_predict not in vocab:
                word_to_predict = unknown_token
            if word_to_predict != prev_word:
                k_most_similar = find_k_most_similar(prev_word, dictionary, k, measure)
                words = [el[0] for el in k_most_similar]
                similarities = [el[1] for el in k_most_similar]
                similarities = [(float(i) - min(similarities)) / (max(similarities) - min(similarities))
                                for i in similarities]
                if word_to_predict not in words:
                    print("Word", word_to_predict, "not within", k, "most similar words to", prev_word)
                    probabilities.append(0.00000000001)
                else:
                    total_sum = 0
                    for sim in similarities:
                        total_sum = total_sum + sim
                    ind = words.index(word_to_predict)
                    prob = similarities[ind] / total_sum
                    if prob != 0:  # fixing a weird bug here
                        probabilities.append(prob)
                counter = counter + 1
        prev_word = word_to_predict
        if counter > max_length:
            break

    n = len(probabilities)
    if n == 0:
        return float('inf')
    else:  # help from https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954
        language = 0
        for prob in probabilities:
            language = language + np.log2(prob)
        language = language / n
        return np.power(2, -language)


def calculate_spearman(rankings1, rankings2):
    """
    :param rankings1: list containing items
    :param rankings2: list containing same items as rankings1, but ranked differently
    :return: spearman rank correlation coefficient between rankings1 and rankings2
    """
    # using https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    score = 0
    n = len(rankings1)
    for ranking in rankings1:
        rank1 = rankings1.index(ranking)
        rank2 = rankings2.index(ranking)
        score = score + ((rank1 - rank2)**2)
    return 1 - ((6 * score) / (n * (n**2 - 1)))


def calculate_bigram_correlation(text, dictionary, measure="cos"):
    """
    :param text: string to evaluate
    :param dictionary: of form {word: vector}
    :param measure: "cos" for cosine similarity metric, "euclidean" for euclidean distance
    :return: bi-gram correlation between -1 and 1
    """
    text = tokenizer.basic(text)
    bigrams = list(nltk.bigrams(text))
    bigram_dict = util.create_dict_of_counts(bigrams)
    bigram_dict = {k: v for k, v in sorted(bigram_dict.items(), key=lambda item: item[1], reverse=True)}
    similarity_dict = dict()

    bigram_rankings = []
    for bigram in bigram_dict:
        vector1 = None
        vector2 = None
        word1 = bigram[0]
        word2 = bigram[1]
        try:
            vector1 = dictionary[word1]
        except KeyError:
            try:
                vector1 = dictionary[unknown_token]
            except KeyError:
                pass
        try:
            vector2 = dictionary[word2]
        except KeyError:
            try:
                vector2 = dictionary[unknown_token]
            except KeyError:
                pass
        if vector1 is not None and vector2 is not None:
            if measure == "cos":
                sim = calculate_cosine_similarity(vector1, vector2)
            elif measure == "euclidean":
                sim = -calculate_euclidean_distance(vector1, vector2)  # lower distance == more similar
            else:
                raise RuntimeError("Must use cos or euclidean distance metric")
            if sim != 0:  # making sure we're not comparing 2 UNK
                similarity_dict.update({bigram: sim})
                bigram_rankings.append(bigram)

    similarity_dict = {k: v for k, v in sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)}

    similarity_rankings = list(similarity_dict.keys())
    return calculate_spearman(bigram_rankings, similarity_rankings)
