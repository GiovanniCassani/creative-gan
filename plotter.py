from tensorflow.python.keras.models import load_model
from word2vec.load_w2v import load_embeddings
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_collage(w, generator, word_index, rows=7, cols=7):
    fig, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            image = generate_image(generator, word_index)
            axs[row, col].imshow((image[0, :, :, 0]), cmap='gray')
            axs[row, col].axis('off')
    fig.savefig("data/{}.png".format(w))
    plt.show()
    plt.close()


def generate_image(generator, word_index):

    word_index = np.asarray(word_index).reshape(-1, 1)
    noise = np.random.normal(0, 1, (1, 100))
    image = generator.predict([noise, word_index])
    image = 0.5 * image + 0.5
    return image


def generate_info(model, w):

    vocab = ['shirts', 'trousers', 'pullover', 'dress',
             'coat', 'sandals', 'shirt', 'sneakers', 'bag', 'boots']

    for i in vocab:
        similarity = model.similarity(i, w)
        print('Similarity between {} and {}: {}'.format(w, i, similarity))


if __name__ == '__main__':
    iters = 100000
    print('[*] Loading Models')
    word2vec = load_embeddings('/home/gcassani/Resources/Embeddings/glove.6B/glove.6B.100d.txt')
    generator = load_model('generator_{}_epochs'.format(iters))

    while 1:
        word = input("\n[*] Write a word --> ")
        if word == '':
            sys.exit(0)

        index = word2vec.vocab[word].index

        generate_info(word2vec, word)
        generate_collage(word, generator, index)
