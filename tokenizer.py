import re
import math
import util
import nltk

"""
# Doing this causes a lot of (presumably dependency) errors, so unable to allow these to be user-input
import word2vec
SUBSAMPLING_RATE = word2vec.SUBSAMPLING_RATE
DISCOUNT = word2vec.DISCOUNT
THRESHOLD = word2vec.THRESHOLD
PASSES = word2vec.PASSES
"""
SUBSAMPLING_RATE = 0.001
DISCOUNT = 0.75
THRESHOLD = 0.15
PASSES = 4


def basic(text):
    """
    :param text: a string
    :return: a list of every word in the string, in lower case with all non alpha-numeric characters removed
    """
    text = text.lower()
    regex = re.compile(r'[^A-Za-z0-9 ]+')
    text = regex.sub('', text)
    text = text.split()
    return text


def get_vocab_size(text_list):
    """
    :param text_list: either of the outputs from set_one_to_one_ids method
    :return: number of unique words in the original corpus
    """
    return len(text_list)


def subsampling_of_frequent_words(text, t=SUBSAMPLING_RATE):
    """
    :param text: a string or a list of words
    :param t: how "aggressively" to subsample
    :return: sub-sampled list of words
    """
    # shakespeare: originally 4500 vocab
    #   t=0.0001 leaves 17k corpus and 1700 vocab
    #   t=0.001 leaves 28k corpus and 2800 vocab
    # imdb: similar
    # t=0.00001 suggested in paper (too aggressive though lol)
    if type(text) is str:
        text = basic(text)
    corpus_size = get_vocab_size(text)
    vocab = set(text)
    for word_to_remove in vocab:
        freq = text.count(word_to_remove) / corpus_size
        prob = 1 - math.sqrt(t / freq)
        throwaway = util.decision(prob)
        if throwaway:
            text = list(filter(lambda a: a != word_to_remove, text))
    return text


def learn_phrases(text, discount=DISCOUNT, threshold=THRESHOLD, passes=PASSES):
    """
    :param text: a string or a list of words
    :param discount: to be subtracted from score
    :param threshold: to create single token out of bigram
    :param passes: how many times to construct phrases
    :return: list of words, where frequent phrases have been transformed into a single token
    """
    # for shakespeare: discount=0.75 and threshold=0.15 or 0.2
    # for imdb: same works, with threshold=0.15 better
    for _ in range(passes):
        if type(text) == str:
            text = basic(text)

        bigrams = list(nltk.bigrams(text))
        # bigrams with score above the chosen threshold are then used as phrases
        bigram_dict = util.create_dict_of_counts(bigrams)
        bigrams_to_replace = []
        for key in bigram_dict:
            bigram_count = bigram_dict[key]
            w1, w2 = key
            w1_count = text.count(w1)
            w2_count = text.count(w2)
            # score = [count(w1,w2) - discount] / [count(w1) x count(w2)]
            score = (bigram_count - discount) / (w1_count + w2_count)
            if score >= threshold:
                # bigrams with score above the chosen threshold are then used as phrases
                bigrams_to_replace.append(key)

        string_text = ' '.join(text)
        for bigram in bigrams_to_replace:
            original_bigram_string = bigram[0] + " " + bigram[1]
            new_bigram_string = bigram[0] + bigram[1]
            string_text = string_text.replace(original_bigram_string, new_bigram_string)
        text = string_text

        threshold = threshold * 0.95

    text = basic(text)
    return text


def super_tokenizer(text):
    """
    :param text: a string
    :return: list of words after subsampling and finding phrases
    """
    text = subsampling_of_frequent_words(text)
    text = learn_phrases(text)
    return text


def out_of_vocabulary_smoothing(text):  # TODO: test
    """
    :param text: a string or a list of words in the corpus
    :return: a list of words after replacing "rare" words (with frequency 1) with the UNK token
    """
    if type(text) == str:
        text = basic(text)

    new_text = []
    for word in text:
        freq = text.count(word)
        if freq == 1:
            new_text.append("UNK")
        else:
            new_text.append(word)

    text = new_text
    return text
