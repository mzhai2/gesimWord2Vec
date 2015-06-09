import logging, sys, multiprocessing
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def readLines():
    for f in open(sys.argv[1], 'r'):
        for line in f:
            yield line

model = Word2Vec(readLines(), size=300, window=5, negative=10, sample=.00001, iter=15, workers=multiprocessing.cpu_count())
model.save_word2vec_format(sys.argv[2], binary=False)