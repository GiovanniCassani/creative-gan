from tensorflow.python.keras.models import load_model
from word2vec.load_w2v import load_embeddings
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_collage(w, gen, word_index, rows=2, cols=2):

    """
    :param w:           str, the target word
    :param gen:         the trained generator from a conditional GAN
    :param word_index:  int, indicating the index in the embedding space of the input word
    :param rows:        int, indicating how many rows the output collage will have
    :param cols:        int, indicating how many cols the output collage will have
    :return:
    """

    fig, axs = plt.subplots(rows, cols)
    for row in range(rows):
        for col in range(cols):
            image = generate_image(gen, word_index, batch_size=1)
            if gen.output_shape[-1] == 1:
                axs[row, col].imshow((image[0, :, :, 0]), cmap='gray')
            else:
                axs[row, col].imshow((image[0, :, :, :]))
            axs[row, col].axis('off')

    # saves a .png file in the collages/ dir
    outdir = 'collages/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig("{}/{}.png".format(outdir, w))
    plt.show()
    plt.close()


def generate_image(gen, word_index, batch_size=1):

    """
    :param gen:         the trained generator from a conditional GAN
    :param word_index:  int, indicating the index in the embedding space of the word for which an image is being
                        generated
    :param batch_size:  int, indicating how many images to generate at once. Default=1.
    :return:            an image generated on the basis of the embedding of the word at the input index
    """

    word_index = np.asarray(word_index).reshape(-1, batch_size)
    noise = np.random.normal(0, 1, (batch_size, gen.input_shape[0][1]))
    image = gen.predict([noise, word_index])
    image = 0.5 * image + 0.5
    return image


def retrieve_similarities(embeddings, w, vocab):

    """
    :param embeddings:  a word2vec embedding space
    :param w:           str, indicating a word for which an embedding is available
    :param vocab:       list of strings, indicating the reference vocabulary
    
    Prints the cosine similarity between the input word and each word from the reference vocabulary as
    estimated from the input embedding space
    """

    for i in vocab:
        similarity = embeddings.similarity(i, w)
        print('Similarity between {} and {}: {}'.format(w, i, similarity))


def oov_words(gen, embeddings, vocabulary):

    """
    :param gen:         the trained generator from a conditional GAN
    :param embeddings:  a word2vec embedding space
    :param vocabulary:  list of strings, indicating the training vocabulary

    Creates a collage of images generated for a word passed interactively based on the similarity between the embedding
    of the target word and the embeddings of the words in the reference vocabulary.
    """

    while 1:
        word = input("\n[*] Write a word --> ")
        if word == '':
            sys.exit(0)

        index = embeddings.vocab[word].index

        retrieve_similarities(embeddings, word, vocabulary)
        generate_collage(word, gen, index)


def training_words(gen, discr, training_vocab, n=100):

    """
    :param gen:             the trained generator from a conditional GAN
    :param discr:           the trained discriminator from a conditional GAN
    :param training_vocab:  a list of strings, indicating the training vocabulary
    :param n:               the number of images to generate for each word from the training vocabulary (default=100)
    :return:
    """

    loss, acc = []
    for w in training_vocab:
        w_idx = word_embeddings.vocab[w].index
        fake_imgs = generate_image(gen, w_idx, batch_size=n)
        fake_labels = fake = np.zeros((n, 1))


if __name__ == '__main__':

    iters = 100000
    data = 'cifar'

    labels_path = 'data/cifar100_labels.txt'
    with open(labels_path, 'r') as fin:
        labels = [label.rstrip() for label in fin if not label.startswith('#')]

    print('[*] Loading Models')
    word_embeddings = load_embeddings('/home/gcassani/Resources/Embeddings/glove.6B/glove.6B.100d.txt')
    generator = load_model('models_{}/generator_{}_epochs'.format(data, iters))
    discriminator = load_model('models_{}/discriminator_{}_epochs'.format(data, iters))
