The Shakespeare dataset comes from: 
https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

The IMDb dataset comes from: 
https://ai.stanford.edu/~amaas/data/sentiment/

The Netflix dataset comes from: 
https://github.com/ygterl/EDA-Netflix-2020-in-R/blob/master/netflix_titles.csv

util.py contains just a couple of general functions used by other scripts, such as dealing with dictionaries.

tokenizer.py contains various methods to tokenize the text before training with it. See project report for more details.

w2v_models.py contains the all of the methods necessary to prepare and train the CBOW and Skip-gram models, including training the neural network itself.

word2vec.py is how the user interacts with and specifies conditions for the CBOW and Skip-gram models themselves. See example usages in word2vec_shakespeare.ipynb and word2vec_imdb.ipynb

evaluation_metrics.py contains the methods necessary to evaluate the learned word embeddings. See project report for more details.

evaluate.py is how the user interacts with and specifies the learned word embeddings and file to evaluate on. See example usages in word2vec_evaluation.ipynb

word2vec_application.py is a script which implements a Netflix recommendation system using the Skip-gram model. See training in word2vec_evaluation.ipynb

recommender.py is how the user interacts with and specifies a movie/TV show they enjoy in order to get Netflix recommendations. See example usages in word2vec_evaluation.ipynb