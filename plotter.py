from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import sys


def generate_collage(word, model, word_index, rows=7, cols=7):
    fig, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            image = generate_image(model, word_index)
            axs[row, col].imshow((image[0, :, :, 0]), cmap='gray')
            axs[row, col].axis('off')
    fig.savefig("data/{}.png".format(word))
    plt.show()
    plt.close()


def generate_image(model, word_index):
    word_index = np.asarray(word_index).reshape(-1, 1)
    noise = np.random.normal(0, 1, (1, 100))
    image = generator.predict([noise, word_index])
    image = 0.5 * image + 0.5
    return image


def generate_info(model, word):
    vocab = ['shirts', 'trousers', 'pullover', 'dress',
             'coat', 'sandals', 'shirt', 'sneakers', 'bag', 'boots']

    for i in vocab:
        similarity = model.wv.similarity(i, word)
        print('Similarity between {} and {}: {}'.format(word, i, similarity))


if __name__ == '__main__':
    iters = 70000
    print('[*] Loading Models')
    word2vec = Word2Vec.load('word2vec/gensim-wiki-model')
    generator = load_model('generator_{}_epochs.h5'.format(iters))

    while 1:
        word = input("\n[*] Write a word --> ")
        if word == '':
            sys.exit(0)

        index = word2vec.wv.vocab[word].index

        generate_info(word2vec, word)
        generate_collage(word, generator, index)
