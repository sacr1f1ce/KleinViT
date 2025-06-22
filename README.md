# KleinViT

This repository contains the implementation for "KleinViT," a research project exploring the integration of Klein bottle geometry into Vision Transformers (ViT).

The core hypothesis is that the manifold of high-contrast 3x3 image patches resembles a Klein bottle. By explicitly encoding this geometric prior into the ViT architecture, we aim to improve model performance, data efficiency, and robustness to transformations like rotation.

This research is based on the articles provided in the `/articles` directory and the step-by-step plan in `main.md`.

## Installation

It is recommended to use `uv` for package management due to its speed. You can install it and set up the environment as follows:

```bash
# First, install uv itself (if you don't have it)
pip install uv

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the project dependencies using uv
uv pip install -r requirements.txt
```

## Repository Structure

```
KleinViT/
├── README.md                # This file
├── main.md                  # Detailed project plan and experimental design
├── articles/                # Relevant research papers
├── data/                    # For storing datasets (e.g., CIFAR-10)
├── notebooks/
│   └── visualization.ipynb  # Notebook for visualizing patch projections
├── output/                  # For saving model outputs and visualizations
├── requirements.txt         # Project dependencies
├── scripts/
│   └── run_training.sh      # Example training script
└── src/
    ├── klein_tools/         # Tools for Klein bottle feature extraction (Gabor filters, etc.)
    ├── model/               # ViT and KleinViT model definitions
    ├── training/            # Training, evaluation, and experiment management scripts
    └── utils.py             # General utility and visualization functions
```