import logging, sys, multiprocessing
from gensim.models import Word2Vec


class sentenceProvider(object):
    def __init__(self, files):
        self.files = files
    def __iter__(self):
        for file in self.files:
            for line in open(file):
                yield line.split()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = sentenceProvider(sys.argv[1:-1])
model = Word2Vec(sentences, size=300, window=5, workers=multiprocessing.cpu_count())
model.accuracy('/home/mzhai/projects/cluster/word2vec/question-phrases')
model.save('/home/mzhai/projects/cluster/output/' + sys.argv[-1:])

# model.load("NYT_GIGA")
# bigram_transformed = gensim.models.Phrases(sentences)
# model = Word2Vec(bigram_transformed[sentences], size=100,)

