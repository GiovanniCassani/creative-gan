from tensorflow.python.keras.datasets import fashion_mnist as mnist
from tensorflow.python.keras.datasets import cifar100 as cifar
from tensorflow.python.keras.layers import Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.python.keras.layers import Input, BatchNormalization,  Embedding
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from word2vec.load_w2v import load_embeddings
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import configparser

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CGAN:

    def __init__(self, config_options):

        """
        :param config_options:     config parser object
        """

        # Images
        self.img_dataset = config_options['Images']['dataset']
        self.img_rows = int(config_options['Images']['rows'])
        self.img_cols = int(config_options['Images']['cols'])
        self.channels = int(config_options['Images']['channels'])
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.output_image_grid = [int(el) for el in config_options['Images']['out_grid'].split(', ')]

        # Word embeddings
        self.word_embeddings = load_embeddings(config_options['Embeddings']['file'])
        self.embeddings_dim = self.word_embeddings.vector_size
        self.embeddings_vocab = len(self.word_embeddings.vocab)
        self.train_labels = self.read_labels(
            os.path.join(os.getcwd(), 'data', config_options['Embeddings']['train_vocab'])
        )
        self.test_words = config_options['Embeddings']['test_vocab'].split(', ')
        _, self.index = self.build_index()

        # GAN
        optimizer = Adam(0.0002, 0.5)   # the first value is the learning rate and the second is the beta1, right?

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(config_options['Discriminator'])
        self.discriminator.compile(loss=config_options['Discriminator']['loss'],
                                   optimizer=optimizer,
                                   metrics=config_options['Discriminator']['metrics'].split(', '))

        # Build the generator
        self.generator = self.build_generator(config_options['Generator'])

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.embeddings_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discr takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.gan = Model([noise, label], valid)
        self.gan.compile(loss=config_options['GAN']['loss'], optimizer=optimizer)

        print('Discriminator summary:')
        self.discriminator.summary()

        print('\nGenerator summary:')
        self.generator.summary()

        print('\nGAN summary:')
        self.gan.summary()

        self.epochs = int(config_options['GAN']['epochs'])
        self.batch_size = int(config_options['GAN']['batch_size'])
        self.sample_interval = int(config_options['GAN']['sample_interval'])
        self.image_folder = config_options['GAN']['image_folder']
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self.outfolder = config_options['GAN']['outfolder']
        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)

    def read_labels(self, path):

        with open(path, 'r') as fin:
            labels = [w.strip() for w in fin if not w.startswith('#')]
        return labels

    def build_index(self, training=True):

        target_words = deepcopy(self.train_labels)

        if not training:
            for w in self.test_words:
                target_words.append(w)

        indexed_words = []
        for word in target_words:
            indexed_words.append(self.word_embeddings.vocab[word].index)

        return target_words, indexed_words

    """
    Questa mi pare non serva e non capisco cosa debba fare:
    
    def build_weights(self):

        weight_matrix = np.zeros((len(self.vocab), 100))
        # step vocab, store vectors using the Tokenizer's integer mapping
        for i, word in enumerate(self.vocab):
            weight_matrix[i] = self.word2vec['word']
        return weight_matrix
    """

    def build_generator(self, config_gener):

        m = float(config_gener['momentum'])
        a = float(config_gener['leaky_relu_alpha'])

        model = Sequential()

        model.add(Dense(int(config_gener['dense1']), input_dim=self.embeddings_dim))
        model.add(LeakyReLU(alpha=a))
        model.add(BatchNormalization(momentum=m))
        model.add(Dense(int(config_gener['dense2'])))
        model.add(LeakyReLU(alpha=a))
        model.add(BatchNormalization(momentum=m))
        model.add(Dense(int(config_gener['dense3'])))
        model.add(LeakyReLU(alpha=a))
        model.add(BatchNormalization(momentum=m))
        model.add(Dense(np.prod(self.img_shape), activation=config_gener['activation']))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.embeddings_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(
            self.embeddings_vocab, self.embeddings_dim, weights=[self.word_embeddings.vectors], trainable=False
        )(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self, config_discr):

        a = float(config_discr['leaky_relu_alpha'])
        d = float(config_discr['dropout'])

        model = Sequential()

        model.add(Dense(int(config_discr['dense1']), input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=a))
        model.add(Dense(int(config_discr['dense2'])))
        model.add(LeakyReLU(alpha=a))
        model.add(Dropout(d))
        model.add(Dense(int(config_discr['dense3'])))
        model.add(LeakyReLU(alpha=a))
        model.add(Dropout(d))
        model.add(Dense(1, activation=config_discr['activation']))

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(
            self.embeddings_vocab, self.embeddings_dim, weights=[self.word_embeddings.vectors], trainable=False
        )(label))
        label_expansion = Dense(self.img_rows*self.img_cols*self.channels)(label_embedding)
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_expansion])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self):

        # Load the dataset
        if 'mnist' in self.img_dataset:
            (X_train, y_train), (_, _) = mnist.load_data()
        elif 'cifar' in self.img_dataset:
            (X_train, y_train), (_, _) = cifar.load_data('fine')
            X_train = X_train / 255.0
            classes2labels = {c: l for c, l in enumerate(self.train_labels)}
            labels2classes = {v: k for k, v in classes2labels.items()}
        else:
            raise ValueError(
                "Unknown dataset {}! Please use either 'fasion_mnist' or 'cifar100'.".format(self.img_dataset)
            )

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # if self.channels == 1:
        #     X_train = np.expand_dims(X_train, axis=3)

        for n, i in enumerate(self.index):
            y_train = np.where(y_train == n, i, y_train)

        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (self.batch_size, self.embeddings_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator; it's possible to assign a different weight to the loss for real images v. that
            # for generated images
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.choice(self.index, self.batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.gan.train_on_batch([noise, sampled_labels], valid)

            # If at save interval => save generated image samples
            if epoch % self.sample_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                self.sample_images(epoch)

        self.save()

    def save(self):

        self.discriminator.trainable = False
        self.gan.save(os.path.join(self.outfolder, 'gan_{}_epochs'.format(self.epochs)))
        self.discriminator.trainable = True
        self.discriminator.save(os.path.join(self.outfolder, 'discriminator_{}_epochs'.format(self.epochs)))
        self.generator.save(
            os.path.join(self.outfolder, 'generator_{}_epochs'.format(self.epochs)), include_optimizer=False
        )

    def sample_images(self, epoch):

        r, c = self.output_image_grid
        vocab, extended_index = self.build_index(training=False)
        noise = np.random.normal(0, 1, (r * c, self.embeddings_dim))
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
        fig.savefig("{}/{}.png".format(self.image_folder, epoch))
        plt.close()


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('GANconfig_mnist.ini')
    cgan = CGAN(config)
    cgan.train()
