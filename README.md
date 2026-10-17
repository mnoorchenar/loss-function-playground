---
title: ML Loss Function Playground
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<h1>🧠 ML Loss Function Playground</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=7c3aed&center=true&vCenter=true&width=700&lines=Compare+Loss+Functions+Side+by+Side;Regression+·+Binary+·+Multi-Class;Train+Live+with+PyTorch" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/loss-functions)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/loss-functions)

<br/>

**🧠 ML Loss Function Playground** — An interactive web application for exploring and comparing loss functions in regression and classification tasks using **PyTorch**.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Technical Overview](#-technical-overview)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [How It Works](#️-how-it-works)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

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

---

## 🧰 Technical Overview

- **Framework**: Flask + PyTorch  
- **Frontend**: HTML + Plotly.js  
- **Dataset Generation**: `scikit-learn` synthetic datasets  
- **Metrics**:
  - Regression: MSE, MAE, R²
  - Classification: Accuracy, Precision, Recall, F1

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/loss-functions.git
cd loss-functions

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
docker build -t loss-functions .
docker run -p 7860:7860 loss-functions
```

---

## 🏗️ How It Works

1. The app generates a synthetic dataset for the chosen task.  
2. For each selected loss function:
   - It builds a simple fully connected PyTorch network.
   - Trains it for the specified number of epochs.
   - Tracks loss, convergence, and timing.
3. The results are visualized interactively in the browser.

---

## 📁 Project Structure

```
loss-functions/
│
├── 📄 Dockerfile                 # Container definition (port 7860)
├── 📄 app.py                     # Application entry point
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available. This software is provided "as is" without warranty of any kind; use at your own risk.

---

## 📜 License

© 2025 Mohammad Noorchenarboo. All rights reserved.

This project and all associated code are the intellectual property of the author. Reproduction, redistribution, or modification of any part of this repository is strictly prohibited without prior written permission. For permissions or inquiries, contact: [LinkedIn](https://www.linkedin.com/in/mnoorchenar)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7c3aed,100:06b6d4&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![Star on GitHub](https://img.shields.io/badge/%E2%AD%90_Star-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/loss-functions/stargazers)
[![Fork on GitHub](https://img.shields.io/badge/%F0%9F%8D%B4_Fork-on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/loss-functions/fork)

<sub>The name "ML Loss Function Playground" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>
