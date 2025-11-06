---
title: ML Loss Function Playground
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 ML Loss Function Playground

An interactive web application for exploring and comparing loss functions in regression and classification tasks using **PyTorch**.

## Features

- Compare multiple loss functions side by side:
  - MSE, MAE, Huber (for regression)
  - BCE, Cross Entropy, Sparse Categorical Cross Entropy (for classification)
- Switch between:
  - Regression
  - Binary classification
  - Multi-class classification
- Generate synthetic datasets with adjustable:
  - Noise level
  - Outlier percentage and magnitude
  - Number of samples and features
- Visualize:
  - Training vs validation loss curves
  - Total training time per loss function
  - Metric tables (Accuracy, F1, R², etc.)
- View convergence epochs and speed comparisons

## Technical Overview

- **Framework**: Flask + PyTorch  
- **Frontend**: HTML + Plotly.js  
- **Dataset Generation**: `scikit-learn` synthetic datasets  
- **Metrics**:
  - Regression: MSE, MAE, R²
  - Classification: Accuracy, Precision, Recall, F1  

## How It Works

1. The app generates a synthetic dataset for the chosen task.  
2. For each selected loss function:
   - It builds a simple fully connected PyTorch network.
   - Trains it for the specified number of epochs.
   - Tracks loss, convergence, and timing.
3. The results are visualized interactively in the browser.


## Credits and Copyright

© 2025 Mohammad Noorchenarboo. All rights reserved.

This project and all associated code are the intellectual property of the author.  
Reproduction, redistribution, or modification of any part of this repository is strictly prohibited without prior written permission.

For permissions or inquiries, contact: [LinkedIn](https://www.linkedin.com/in/mnoorchenar)

