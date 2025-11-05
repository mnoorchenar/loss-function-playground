---
title: VAE Interactive Playground
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 VAE Interactive Playground

An interactive web application for training and visualizing Variational Autoencoders (VAE) on the MNIST dataset.

## Features

- **Training Dashboard**: Configure and train your VAE with custom hyperparameters
- **Architecture Visualization**: Understand the VAE architecture and components
- **Latent Space Explorer**: Visualize the 2D latent space representation
- **Reconstruction**: See how well the VAE reconstructs MNIST digits
- **Generation**: Generate new digits by manipulating latent dimensions

## Usage

1. Navigate to the **Training Dashboard** tab
2. Configure your VAE parameters (epochs, batch size, learning rate, hidden dimensions, latent dimensions)
3. Click **Start Training** to train the model
4. Once training is complete, explore the other tabs to visualize results

## Architecture

The VAE consists of:
- **Encoder**: Maps 784D input (28×28 images) to a lower-dimensional latent space
- **Latent Space**: Learns mean (μ) and variance (σ²) for each latent dimension
- **Decoder**: Reconstructs images from latent representations

## Technical Details

- Framework: Flask + PyTorch
- Dataset: MNIST (10,000 sample subset for faster training)
- Loss: Binary Cross-Entropy + KL Divergence