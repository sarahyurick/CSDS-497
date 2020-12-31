import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.models import Sequential
# from keras.layers import Dense
import tokenizer
# from tensorflow.keras.callbacks import EarlyStopping
import random


def set_one_to_one_ids(tokens_list):
    """
    :param tokens_list: the corpus in list form, after it has been tokenized
    :return: 2 dicts. first has entries of form {word: id} and second has entries of form {id: word}
    """
    lookup_by_word = dict()
    current_id = 0
    for token in tokens_list:
        if token not in lookup_by_word:
            lookup_by_word.update({token: current_id})
            current_id += 1

    lookup_by_id = {v: k for k, v in lookup_by_word.items()}

    return lookup_by_word, lookup_by_id


def generate_center_word_context_pairs(tokens_list, window_size, randomness):
    """
    :param tokens_list: the corpus in list form, after it has been tokenized
    :param window_size: number of words before and words after that are considered to be in the "context" of a word
    :param randomness: if True, for each training word we will select randomly a number R in range <1, window_size>
    :return: a list of tuples of the form (word, [prev_word_w, ..., prev_word_1, after_word_1, ..., after_word_w]
    """
    word_and_context = []
    corpus_size = len(tokens_list)

    for center_word_index in range(corpus_size):
        current_context_words = []
        center_word = tokens_list[center_word_index]
        if randomness:
            window_size = random.randint(1, window_size)
        for j in range(center_word_index - window_size, center_word_index + window_size + 1):
            if j != center_word_index and j >= 0:
                try:
                    context_word = tokens_list[j]
                    current_context_words.append(context_word)
                except IndexError:
                    pass
        word_and_context.append((center_word, current_context_words))

    return word_and_context


def create_one_hot_encodings(word_and_context, lookup_by_word):
    """
    :param word_and_context: output from generate_center_word_context_pairs method
    :param lookup_by_word: 1st output from set_one_to_one_ids method
    :return: 2 lists of lists of the same dimensions. 1st is one-hot encoded individual words.
             2nd is one-hot encoded context words
    """
    vocab_size = len(lookup_by_word)
    center_words_encodings = []
    context_words_encodings = []

    for pair in word_and_context:
        center_word = pair[0]
        center_id = lookup_by_word[center_word]
        center_temp = np.zeros(vocab_size)
        center_temp[center_id] = 1
        center_words_encodings.append(center_temp)

        context_temp = np.zeros(vocab_size)
        for context_word in pair[1]:
            context_id = lookup_by_word[context_word]
            context_temp[context_id] = 1
        context_words_encodings.append(context_temp)

    return center_words_encodings, context_words_encodings


def initialize_nn_parameters(doc_string, tokenizer_type, window_size, randomness):
    """
    :param doc_string: corpus to train on.
    :param tokenizer_type: method to use when tokenizing words. see tokenizer.py for options.
    :param window_size: number of words before and words after that are considered to be in the "context" of a word
    :param randomness: if True, for each training word we will select randomly a number R in range <1, window_size>
    :return: 2 outputs from create_one_hot_encodings method,
             output from get_vocab_size,
             1st output from set_one_to_one_ids method
    """
    tokenized_doc = tokenizer_type(doc_string)
    lookups, _ = set_one_to_one_ids(tokenized_doc)
    vocab_size = tokenizer.get_vocab_size(lookups)
    words_and_contexts = generate_center_word_context_pairs(tokenized_doc, window_size, randomness)
    center_encodings, context_encodings = create_one_hot_encodings(words_and_contexts, lookups)

    center_encodings = np.array(center_encodings)
    context_encodings = np.array(context_encodings)

    return center_encodings, context_encodings, vocab_size, lookups


def train_model(doc, skipgram, tokenizer_type,
                window_size, vector_length,
                iterations=150, batch_size=10, save_file=None,
                stop_value=200,
                act="softmax", opt='SGD', randomness=False):
    """
    :param doc: corpus to train on.
    :param skipgram: True to train skipgram model, False to train cbow model
    :param tokenizer_type: of the form tokenizer.method_name (see tokenizer.py)
    :param window_size: number of words before and words after that are considered to be in the "context" of a word
    :param vector_length: number of dimensions to represent word vectors as
    :param iterations: epochs for neural net
    :param batch_size: for neural net
    :param save_file: what to save the neural net files as
    :param stop_value: if scores do not improve after stop_value epochs, stop - DISABLED !!
    :param act: activation function to use for second layer of weights (see tf keras documentation for options)
    :param opt: optimization method to use (see tf keras documentation for options)
    :param randomness: if True, for each training word we will select randomly a number R in range <1, window_size>
    :return: dict of form {word: vector_representation}
    """
    X, Y, V, vocab_lookups = initialize_nn_parameters(doc, tokenizer_type, window_size, randomness)
    if not skipgram:
        X, Y = Y, X

    # with help from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    # define the keras model
    model = Sequential()
    model.add(Dense(vector_length, input_dim=V, activation='linear'))
    model.add(Dense(V, activation=act))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # EARLY_STOP_MONITOR = EarlyStopping(monitor='accuracy', patience=stop_value)
    # fit the keras model on the dataset with (1) or without (0) progress bars
    # model.fit(X, Y, epochs=iterations, batch_size=batch_size, callbacks=[EARLY_STOP_MONITOR], verbose=1)
    model.fit(X, Y, epochs=iterations, batch_size=batch_size, verbose=1)

    # evaluate the keras model
    # _, accuracy = model.evaluate(X, Y, verbose=0)
    # print('Accuracy: %.2f' % (accuracy*100))

    # the word vectors are defined to be the learned weights from input to hidden layer
    input_to_hidden_model = Sequential([Dense(vector_length, input_dim=V, activation='linear')])
    # set weights of the first layer
    input_to_hidden_model.set_weights(model.layers[0].get_weights())
    input_to_hidden_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    if save_file is not None:
        model.save(save_file)

    # create a dict of all words in vocab and their learned word vector representation
    word_vectors = dict()
    for i in range(len(vocab_lookups)):
        word = list(vocab_lookups.keys())[i]
        word_id = vocab_lookups[word]
        vocab_vector = np.zeros(V)
        vocab_vector[word_id] = 1
        vocab_vector = [vocab_vector]
        vocab_vector = np.array(vocab_vector)
        word_vector = input_to_hidden_model.predict(vocab_vector)
        word_vectors.update({word: word_vector})

    return word_vectors
