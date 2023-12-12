# GAN Implementations

This repository contains implementations of different types of GANs (Generative Adversarial Networks), including GAN (vanilla), DCGAN (Deep Convolutional GAN), WGAN (Wasserstein GAN), and WGAN-GP (Wasserstein GAN with Gradient Penalty).

## Contents

- GAN: Implementation of the original Generative Adversarial Network
- DCGAN: Implementation of a Generative Adversarial Network with deep convolutional layers
- WGAN: Implementation of Wasserstein Generative Adversarial Network
- WGAN-GP: Implementation of Wasserstein Generative Adversarial Network with Gradient Penalty

## Dataset

You can download the CelebA dataset from the [official CelebA dataset website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for training and testing these GAN models.

For other datasets such as MNIST, they are typically available for download through torch. If you have already downloaded the dataset, please place the data in the './data' directory and set '--download' to _False_ during execution.

## Usage

Each subdirectory contains the implementation and usage instructions for the respective model. Please refer to the corresponding README file for more detailed information.

## License

This project is licensed under the [MIT License](LICENSE).
