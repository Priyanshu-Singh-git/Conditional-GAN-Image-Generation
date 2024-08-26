# Conditional GAN for MNIST Dataset

This repository contains a PyTorch implementation of a Conditional Generative Adversarial Network (cGAN) designed to generate images conditioned on labels, specifically using the MNIST dataset. The model consists of an embedding layer, a generator, and a discriminator, and includes custom weight initialization, learning rate schedulers, and loss tracking.

## Project Structure

- **main.py**: Contains the implementation of the cGAN, including the Embedding, Generator, and Discriminator classes, as well as the training loop.
- **data/**: Directory to store the MNIST dataset.
- **images/**: Directory to store generated images and probability distribution plots.
- **README.md**: Project description and setup instructions.

## Dependencies

To run this project, you need the following libraries:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- TorchVision


# Conditional GAN Model Architecture

## Embedding Layer
The embedding layer maps class labels to a higher-dimensional space, enabling the generator to conditionally generate images based on the input labels.

## Generator
The generator takes a noise vector and a label embedding as input and produces a 28x28 grayscale image. The network architecture is as follows:

- **Input:** Latent vector of size 120
- **Layer 1:** Transposed convolution (`ConvTranspose2d`) with 32 filters, kernel size 4, stride 1, and ReLU activation
- **Layer 2:** Spectral normalization + Transposed convolution with 16 filters, kernel size 4, stride 1, and ReLU activation
- **Layer 3:** Spectral normalization + Transposed convolution with 8 filters, kernel size 4, stride 2, padding 1, and ReLU activation
- **Layer 4:** Spectral normalization + Transposed convolution with 4 filters, kernel size 4, stride 2, padding 1, and ReLU activation
- **Output Layer:** Transposed convolution with 1 filter, kernel size 3, stride 1, padding 1, and Tanh activation to produce a 28x28 grayscale image

## Discriminator
The discriminator is a convolutional neural network that receives the generated image along with its label embedding. The network architecture is as follows:

- **Input:** Concatenation of image and label embedding (channels: 21)
- **Layer 1:** Spectral normalization + Convolutional layer (`Conv2d`) with 2 filters, kernel size 4, stride 2, padding 1, and LeakyReLU activation
- **Layer 2:** Spectral normalization + Convolutional layer with 4 filters, kernel size 4, stride 2, padding 0, and LeakyReLU activation
- **Layer 3:** Spectral normalization + Convolutional layer with 8 filters, kernel size 4, stride 2, padding 0, and LeakyReLU activation
- **Layer 4:** Spectral normalization + Convolutional layer with 16 filters, kernel size 3, stride 2, padding 1, and LeakyReLU activation
- **Fully Connected Layer:** Flatten + Linear layer with 100 units, followed by a Linear layer with 10 units for final classification

## Training
The model is trained for 50 epochs with the following settings:

- **Loss Function:** Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
- **Optimizers:**
  - AdamW optimizer for the discriminator (lr = 0.0003)
  - AdamW optimizer for the generator (lr = 0.0012)
  - AdamW optimizer for the embedding layer (lr = 0.0004)
- **Learning Rate Schedulers:**
  - StepLR for discriminator (step size: 2, gamma: 0.1)
  - StepLR for generator (step size: 1, gamma: 0.1)

During training, the generator and discriminator losses are tracked, and probability distributions of real and generated data are plotted to visualize the training progress.

## Probability Distribution Graphs
Below are the probability distribution graphs showing the distributions of real and generated data at different training epochs.

![Probability Distribution Graph 1](./prob_dist/Figure%202024-07-13%20222956%20(1).png)

![Probability Distribution Graph 2](./prob_dist/Figure%202024-07-13%20222956%20(2).png)

![Probability Distribution Graph 3](./prob_dist/Figure%202024-07-13%20222956%20(4).png)

![Probability Distribution Graph 4](./prob_dist/Figure%202024-07-13%20222956%20(5).png)

![Probability Distribution Graph 5](./prob_dist/Figure%202024-07-13%20222956%20(7).png)

![Probability Distribution Graph 6](./prob_dist/Figure%202024-07-13%20222956%20(8).png)

![Probability Distribution Graph 7](./prob_dist/Figure%202024-07-13%20222956%20(10).png)

![Probability Distribution Graph 8](./prob_dist/Figure%202024-07-13%20222956%20(11).png)

![Probability Distribution Graph 9](./prob_dist/Figure%202024-07-13%20222956%20(13).png)

![Probability Distribution Graph 10](./prob_dist/Figure%202024-07-13%20222956%20(14).png)

## Generated Images
Below are the generated images produced by the model at different training epochs.

![Generated Image 1](./images_generated/inital/Figure%202024-07-07%20151714%20(0).png)

![Generated Image 2](./images_generated/inital/Figure%202024-07-07%20151714%20(1).png)

![Generated Image 3](./images_generated/inital/Figure%202024-07-07%20151714%20(2).png)

![Generated Image 4](./images_generated/inital/Figure%202024-07-07%20151714%20(3).png)

![Generated Image 5](./images_generated/inital/Figure%202024-07-07%20151714%20(4).png)

![Generated Image 6](./images_generated/inital/Figure%202024-07-07%20151714%20(5).png)

![Generated Image 7](./images_generated/inital/Figure%202024-07-07%20151714%20(6).png)

![Generated Image 8](./images_generated/inital/Figure%202024-07-07%20151714%20(7).png)

![Generated Image 9](./images_generated/inital/Figure%202024-07-07%20151714%20(8).png)

![Generated Image 10](./images_generated/inital/Figure%202024-07-07%20151714%20(9).png)

