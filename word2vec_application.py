import pandas as pd
import re
import random
from w2v_models import set_one_to_one_ids
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import util
import json

vector_length = 100
act = "softmax"
opt = "SGD"
iterations = 1000
batch_size = 10
save_file = "models\\netflix_vectors"
json_file = "vectors\\netflix_vectors.json"

""""
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
"""

data = pd.read_csv("netflix_titles.csv")

"""
Index(['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added',
       'release_year', 'rating', 'duration', 'listed_in', 'description'],
      dtype='object')
      
type: TV Show or Movie
rating: TV-MA, etc.
listed_in: Stand-Up Comedy, etc.
"""

data = data.drop(['show_id', 'type', 'date_added', 'duration'], axis=1)
data = data.fillna('')
# center word is TITLE - data['title']
center_words = data['title'].str.replace(' ', '')
# context is everything else
context_df = data.drop(['title'], axis=1)
context_df['director'] = context_df['director'].str.replace(' ', '')
context_df['director'] = context_df['director'].str.replace(',', ' ')
context_df['cast'] = context_df['cast'].str.replace(' ', '')
context_df['cast'] = context_df['cast'].str.replace(',', ' ')
context_df['country'] = context_df['country'].str.replace(' ', '')
context_df['country'] = context_df['country'].str.replace(',', ' ')
context_df['rating'] = context_df['rating'].str.replace(' ', '')
context_df['listed_in'] = context_df['listed_in'].str.replace(' ', '')
context_df['listed_in'] = context_df['listed_in'].str.replace(',', ' ')

context_df['string'] = context_df['director'] + " " + context_df['cast'] + " " + context_df['country'] + \
                       " " + context_df['release_year'].astype(str) + " " + context_df['rating'] + " " + \
                       context_df['listed_in'] + " " + context_df['description']

context_corpus = ' '.join(context_df['string'].tolist())
regex = re.compile(r'[^A-Za-z0-9 ]+')
context_corpus = regex.sub('', context_corpus)
# text_tokens = word_tokenize(context_corpus)
# tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
# context_corpus = ' '.join(tokens_without_sw)

# create entire corpus
title_corpus = ' '.join(center_words.tolist())
total_corpus = context_corpus + " " + title_corpus
total_corpus = total_corpus.lower()
total_corpus = total_corpus.split()
random.shuffle(total_corpus)
# give unique ID to each token in the vocabulary
lookups, _ = set_one_to_one_ids(total_corpus)
vocab_size = len(lookups)

# create list of tuples of form (title, all other words associated with the title)
word_context_pairs = []
for row in range(len(center_words)):
    center_word = center_words[row].lower()
    # center_word = regex.sub('', center_word)
    context = context_df['string'][row].lower()
    context = regex.sub('', context)
    word_context_pairs.append((center_word, context))

# one-hot encoded titles and context words
vocab_size = len(lookups)
center_words_encodings = []
context_words_encodings = []

# slightly modified from create_one_hot_encodings
for pair in word_context_pairs:
    # print(pair)
    center_word = pair[0]
    try:
        center_id = lookups[center_word]
        center_temp = np.zeros(vocab_size)
        center_temp[center_id] = 1
        center_words_encodings.append(center_temp)

        context_temp = np.zeros(vocab_size)
        for context_word in pair[1].split():
            try:
                context_id = lookups[context_word]
                context_temp[context_id] = 1
            except KeyError:
                # print(context_word)
                pass
        context_words_encodings.append(context_temp)
    except KeyError:
        # print(center_word)
        pass

title_encodings, context_encodings = center_words_encodings, context_words_encodings
title_encodings = np.array(title_encodings)
context_encodings = np.array(context_encodings)

X, Y = title_encodings, context_encodings  # skip-gram
# Y, X = title_encodings, context_encodings  # cbow
V = vocab_size

# with help from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# define the keras model
model = Sequential()
model.add(Dense(vector_length, input_dim=V, activation='linear'))
model.add(Dense(V, activation=act))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# fit the keras model on the dataset with (1) or without (0) progress bars
model.fit(X, Y, epochs=iterations, batch_size=batch_size, verbose=1)

# the word vectors are defined to be the learned weights from input to hidden layer
input_to_hidden_model = Sequential([Dense(vector_length, input_dim=V, activation='linear')])
# set weights of the first layer
input_to_hidden_model.set_weights(model.layers[0].get_weights())
input_to_hidden_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

if save_file is not None:
    model.save(save_file)

# create a dict of all words in vocab and their learned word vector representation
word_vectors = dict()
for i in range(len(lookups)):
    word = list(lookups.keys())[i]
    word_id = lookups[word]
    vocab_vector = np.zeros(V)
    vocab_vector[word_id] = 1
    vocab_vector = [vocab_vector]
    vocab_vector = np.array(vocab_vector)
    word_vector = input_to_hidden_model.predict(vocab_vector)
    word_vectors.update({word: word_vector})

word_vectors = util.dict_of_array_values_to_lists(word_vectors)
with open(json_file, 'w') as fp:
    json.dump(word_vectors, fp)
