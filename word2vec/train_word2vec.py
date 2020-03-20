from gensim.models.word2vec import Word2Vec
import os
import time
import string

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.table = str.maketrans('', '', string.punctuation)
 
    def __iter__(self):
        for line in open(self.dirname):
            words = line.split()
            clean_words = [w.translate(self.table) for w in words]
            low_words = [word.lower() for word in clean_words]
            yield low_words
 

if __name__ == "__main__":

    sentences = MySentences('data/small_dataset.txt') # a memory-friendly iterator

    print("Loading Model")
    start = time.time()
    model = Word2Vec(sentences)
    end = time.time()
    print(end - start, ' seconds elapsed')

    print('Saving Model')
    model.save('gensim-wiki-model')
