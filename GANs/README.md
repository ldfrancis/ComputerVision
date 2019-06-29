# Generative Adversarial Networks(GANs)

A GAN is a generative model that is built by making two networks compete. One network, called a generator,  tries to generate data while the other,  called a discriminator, try to flag the generated data as fake data. The goal is for the generator to learn to fool the discriminator into flagging the generated data as real.

The gans package has fuctionalities for building and training a GAN that can learn to generate images from the MNIST dataset. Inspired by the first GAN [paper](https://arxiv.org/abs/1406.2661) by Ian Goodfellow

## usage:
To run a simple GAN training, use this on the terminal

`python -m gans`

This trains the GAN for 60000 epochs. Results of generated images from sample epochs are saved in the folder,` data/checkpoints`

From the results obtained by experimenting with this code, it is only after training for up to 40000 epochs on the MNIST data that the model starts to generate reasonable digits.
