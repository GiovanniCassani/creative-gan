# creative_gan


## What it does

The model takes a word as input and gives back an image representing that word.  

#### The objective
is being able to give the model a new word (a word of which he never saw a representation) and having it to return something interesting

## How it does it

The model consist of two parts:

- a **Generator** that produces images
- a **Discriminator** that classifies images

The Discriminator distinguishes **real** from **fake** images of a given set, like birds, or tables.  
The Generator tries to create images, and is given a **positive feedback** when the image it created is classified as real from the Discriminator.  

The more positive feedback the Generator recieves, the more it is able to create images that resemble the real ones.

Both the Generator and the Discriminator recieve 2 inputs:

- The Generator recieves a noise array + a word embedding
- The Discriminator recieves an image + the word embedding of the image's label

In this way the Generator can be given a word embedding to tell him what to produce, and the discriminator knows what to expect and classify.

## Where it's trained

I decided to train the model on the [fashion mnist dataset](https://github.com/zalandoresearch/fashion-mnist), a dataset of black and white, 28x28 images in 10 categories:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The embedder model is a Word2Vec model trained on 200k articles of Wikipedia, and I used the library [gensim](https://github.com/RaRe-Technologies/gensim) for that.

## How it performs

It can take as input any word, and will return images of the most similar words he learnt to represent.

For example
