from w2v_models import *
import argparse
import json
import util
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-model", required=True)
parser.add_argument("-file", required=False)
parser.add_argument("-folder", required=False)
parser.add_argument("-embedding_length", required=False, type=int)
parser.add_argument("-window_size", required=False, type=int)
parser.add_argument("-iterations", required=False, type=int)
parser.add_argument("-batch_size", required=False, type=int)
parser.add_argument("-tokenizer", required=False)
parser.add_argument("-stop_value", required=False, type=int)
parser.add_argument("-max_chars", required=False, type=int)
parser.add_argument("-activation", required=False)
parser.add_argument("-optimizer", required=False)
parser.add_argument("-randomness", required=False)

"""
# Causes error
parser.add_argument("-subsampling_rate", required=False, type=float)
parser.add_argument("-discount", required=False, type=float)
parser.add_argument("-threshold", required=False, type=float)
parser.add_argument("-passes", required=False, type=int)

SUBSAMPLING_RATE = options.subsampling_rate
DISCOUNT = options.discount
THRESHOLD = options.threshold
PASSES = options.passes

if SUBSAMPLING_RATE is None:
    SUBSAMPLING_RATE = 0.001
if DISCOUNT is None:
    DISCOUNT = 0.75
if THRESHOLD is None:
    THRESHOLD = 0.15
if PASSES is None:
    PASSES = 4
"""

options = parser.parse_args()
model = options.model.lower()
file = options.file
folder = options.folder
VECTOR_LENGTH = options.embedding_length
WINDOW_SIZE = options.window_size
ITERATIONS = options.iterations
BATCH_SIZE = options.batch_size
tokenizer_method = options.tokenizer
STOP_VALUE = options.stop_value
MAX_CHARS = options.max_chars
ACTIVATION = options.activation
OPTIMIZER = options.optimizer
randomness = options.randomness

if VECTOR_LENGTH is None:
    VECTOR_LENGTH = 100
if WINDOW_SIZE is None:
    WINDOW_SIZE = 10
if ITERATIONS is None:
    ITERATIONS = 1000
if BATCH_SIZE is None:
    BATCH_SIZE = 10
if tokenizer_method is None:
    tokenizer_method = "basic"
if STOP_VALUE is None:
    STOP_VALUE = 200
if MAX_CHARS is None:
    MAX_CHARS = 250000
if ACTIVATION is None:
    ACTIVATION = "softmax"
if OPTIMIZER is None:
    OPTIMIZER = "SGD"
if randomness is None:
    randomness = False
else:
    randomness = True

if model == "skipgram" or model == "skip-gram":
    SKIPGRAM = True
elif model == "cbow":
    SKIPGRAM = False
else:
    raise RuntimeError("word2vec model must be skipgram or cbow")

if file is not None:
    with open(file, 'r') as f:
        # shakespeare has 1,115,393 characters
        data = f.read()[:MAX_CHARS].replace('\n', ' ')
elif folder is not None:
    # imdb has 66,485,050 chars
    data = ""
    for files in glob.glob(folder + "*.txt"):
        infile = open(files)
        try:
            a = infile.readline()
            data = data + a + " "
        except UnicodeDecodeError:
            pass
        if len(data) > MAX_CHARS:
            break
        infile.close()
else:
    raise RuntimeError("Must input a file or folder")

if file is not None:
    save_name = file.replace(".txt", "")
else:
    save_name = folder
model_info_string = model + "_" + save_name + "_" + tokenizer_method + "_window" + str(WINDOW_SIZE) + "_chars" +\
                    str(MAX_CHARS) + "_" + ACTIVATION + "_" + OPTIMIZER
if randomness:
    model_info_string = model_info_string + "_rand"
model_file = "models\\" + model_info_string
json_file = "vectors\\" + model_info_string + ".json"
# cbow_word_vectors = train_model(data, False, tokenizer.basic, WINDOW_SIZE, VECTOR_LENGTH)
# skipgram_word_vectors = train_model(data, SKIPGRAM, tokenizer.basic_tokenize, WINDOW_SIZE, VECTOR_LENGTH)
if "subsampling" in tokenizer_method:
    word_vectors = train_model(data, SKIPGRAM, tokenizer.subsampling_of_frequent_words,
                               WINDOW_SIZE, VECTOR_LENGTH,
                               ITERATIONS, BATCH_SIZE, model_file,
                               STOP_VALUE, ACTIVATION, OPTIMIZER, randomness)
elif "phrases" in tokenizer_method:
    word_vectors = train_model(data, SKIPGRAM, tokenizer.learn_phrases,
                               WINDOW_SIZE, VECTOR_LENGTH,
                               ITERATIONS, BATCH_SIZE, model_file,
                               STOP_VALUE, ACTIVATION, OPTIMIZER, randomness)
elif "super_tokens" in tokenizer_method:
    word_vectors = train_model(data, SKIPGRAM, tokenizer.super_tokenizer,
                               WINDOW_SIZE, VECTOR_LENGTH,
                               ITERATIONS, BATCH_SIZE, model_file,
                               STOP_VALUE, ACTIVATION, OPTIMIZER, randomness)
elif "oov" in tokenizer_method:
    word_vectors = train_model(data, SKIPGRAM, tokenizer.out_of_vocabulary_smoothing,
                               WINDOW_SIZE, VECTOR_LENGTH,
                               ITERATIONS, BATCH_SIZE, model_file,
                               STOP_VALUE, ACTIVATION, OPTIMIZER, randomness)
else:
    word_vectors = train_model(data, SKIPGRAM, tokenizer.basic,
                               WINDOW_SIZE, VECTOR_LENGTH,
                               ITERATIONS, BATCH_SIZE, model_file,
                               STOP_VALUE, ACTIVATION, OPTIMIZER, randomness)

word_vectors = util.dict_of_array_values_to_lists(word_vectors)

with open(json_file, 'w') as fp:
    json.dump(word_vectors, fp)
