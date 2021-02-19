from tensorflow.python.keras.datasets import fashion_mnist as mnist
from tensorflow.python.keras.layers import Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.python.keras.layers import Input, BatchNormalization,  Embedding
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from word2vec.load_w2v import load_embeddings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os


class CGAN():
    
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.word2vec = load_embeddings(
            '/home/gcassani/Resources/Embeddings/glove.6B/glove.6B.100d.txt'
        )
        self.embedded_dimension = len(self.word2vec.wv.vocab)
        _, self.index = self.build_index()

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discr takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

        print('Discriminator summary:')
        self.discriminator.summary()

        print('\nGenerator summary:')
        self.generator.summary()

        print('\nGAN summary:')
        self.combined.summary()

    def build_index(self, training=True):

        vocab = ['shirts', 'trousers', 'pullover', 'dress',
                 'coat', 'sandals', 'shirt', 'sneakers', 'bag', 'boots']
        if not training:
            vocab.append('sweater')
            vocab.append('kimono')

        indexed_words = []
        for word in vocab:
            indexed_words.append(self.word2vec.wv.vocab[word].index)

        return vocab, indexed_words

    """
    def build_weights(self):

        weight_matrix = np.zeros((len(self.vocab), 100))
        # step vocab, store vectors using the Tokenizer's integer mapping
        for i, word in enumerate(self.vocab):
            weight_matrix[i] = self.word2vec['word']
        return weight_matrix
    """

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.embedded_dimension,
                                              self.latent_dim,
                                              weights=[
                                                  self.word2vec.wv.vectors],
                                              trainable=False)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.embedded_dimension,
                                              self.latent_dim,
                                              weights=[
                                                  self.word2vec.wv.vectors],
                                              trainable=False)(label))
        label_expansion = Dense(784)(label_embedding)
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_expansion])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        for n, i in enumerate(self.index):
            y_train = np.where(y_train == n, i, y_train)

        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                [imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch(
                [gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.choice(self.index,
                                              batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch(
                [noise, sampled_labels], valid)

            # If at save interval => save generated image samples and print progress
            if epoch % sample_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                self.sample_images(epoch)
        self.generator.save('generator_{}_epochs.h5'.format(epochs))

    def sample_images(self, epoch):
        r, c = 2, 6
        vocab, extended_index = self.build_index(training=False)
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.asarray(extended_index).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title(vocab[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images_training/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=1000, batch_size=32, sample_interval=200)
