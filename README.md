# creative_gan

## What it does

The model takes a word as input and gives back an image representing that word.

## How it does it

The model consist of two parts:

- a **Generator** that produces images
- a **Discriminator** that classifies images

The Discriminator distinguishes **real** from **fake** images of a given set, like birds, or tables.  
The Generator tries to create images, and is given a **positive feedback** when the image he created is classified as real from the Discriminator.  

The more positive feedback the Generator recieves, the more it understand which kind of images are able to fool the Discriminator.  

Both the Generator and the Discriminator recieve 2 inputs:

- The Generator recieves a noise array + a word embedding
- The Discriminator recieves an image + the word embedding of the image's label

In this wa
