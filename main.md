# KleinViT Project Plan

This document outlines the plan for experimenting with Klein bottle geometry in Vision Transformers for image classification. The core idea is that the space of high-contrast 3x3 image patches has a topological structure resembling a Klein bottle, and leveraging this structure within a ViT could lead to improved performance and data efficiency.

## Phase 1: Foundational Setup

### 1.1. Project Structure

A clean and scalable project structure will be created as follows:

```
KleinViT/
├── README.md                # Main project README
├── main.md                  # This project plan
├── articles/                # Research papers
├── data/                    # For storing datasets (e.g., CIFAR-10) - gitignored
├── notebooks/
│   └── visualization.ipynb  # Notebook for visualizing patch projections
├── output/                  # For saving model outputs and visualizations
├── requirements.txt         # Project dependencies
├── scripts/
│   └── run_training.sh      # Example training script
└── src/
    ├── __init__.py
    ├── klein_tools/           # Tools for Klein bottle geometry
    │   ├── __init__.py
    │   └── feature_extractor.py # Gabor filters, coordinate extraction
    ├── model/               # ViT model definitions
    │   ├── __init__.py
    │   └── vit.py             # Standard ViT
    │   └── klein_vit.py       # ViT with Klein modifications
    ├── training/              # Training and evaluation scripts
    │   ├── __init__.py
    │   ├── trainer.py
    │   └── evaluate.py
    └── utils.py                 # Utility and visualization functions
```

### 1.2. Baseline ViT

*   **Goal:** Establish a performance benchmark.
*   **Action:** Implement or use a standard Vision Transformer model.
*   **Dataset:** CIFAR-10.
*   **Details:** Train the standard ViT on CIFAR-10. This will be the baseline against which all modifications are compared.

## Phase 2: Feature Extraction (Klein Bottle Coordinates)

*   **Goal:** Extract geometric coordinates (orientation, phase) for each image patch.
*   **Details:** For each `3x3` patch `p` in an image, we need to calculate its `(θ, φ)` coordinates on the Klein bottle.
    *   **Orientation (θ):** Can be estimated using a set of Gabor filters. The angle of the filter with the maximum response for a patch will be its orientation `θ`. Steerable filters are another option.
    *   **Phase (φ):** This relates to the position of the feature within the patch. It can be derived from the phase component of the Gabor filter response. To begin with, we can simplify by focusing only on orientation `θ`, effectively modeling the feature space on a circle.

*   **Action:** Create a Python module `src/klein_tools/feature_extractor.py` to handle this. It should contain a function that takes an image patch and returns `(θ, φ)`.
*   **Visualization:** Use `notebooks/visualization.ipynb` to visualize the extracted coordinates. Do the projections of patches from natural images look like a Klein bottle?

## Phase 3: Integrating Klein Geometry into ViT

This phase has multiple approaches, ordered by increasing complexity.

### Approach A: Geometric Positional Encoding

This is the simplest modification. Instead of modifying the attention mechanism itself, we just give the model information about the geometric coordinates.

*   **Your New Input:**
    1.  Take the geometric coordinates `gi = (θi, φi)`.
    2.  Convert them into a fixed-size vector. Since they are angles, a good way to do this is `[sin(θi), cos(θi), sin(φi), cos(φi)]`. This respects their circular nature.
    3.  Project this small vector into the main embedding dimension D using a small learnable linear layer. Let's call this the `geometric_content_embedding`.
    4.  Your final input token is: `token = patch_embedding + spatial_positional_embedding + geometric_content_embedding`

### Approach B: Topological Attention

*   **Concept:** Modify the attention scores to incorporate a bias based on the Klein bottle distance between patches.
*   **Implementation:**
    1.  **Define a Klein Bottle Distance:** The distance between two points `g1=(θ1, φ1)` and `g2=(θ2, φ2)` on a Klein bottle is complex. A simple approximation (treating it as a torus, ignoring the twist for now) is:
        *   `d_θ = min(|θ1 - θ2|, 2π - |θ1 - θ2|)` (distance on a circle)
        *   `d_φ = min(|φ1 - φ2|, 2π - |φ1 - φ2|)`
        *   `Distance_squared = d_θ^2 + d_φ^2`
    2.  **Similarity Metric:** `Similarity = exp(-Distance_squared / σ^2)`, where `σ` is a learnable parameter.
    3.  **Modify Attention Formula:**
        *   Calculate standard dot-product attention: `S_dot = QK^T / sqrt(d_k)`.
        *   Calculate a geometric bias matrix `B_geom` where `B_geom[i, j]` is the similarity between patch `i` and patch `j`.
        *   New attention scores: `AttentionScores = softmax(S_dot + α * B_geom)`, where `α` is a learnable scalar.
    4.  **The "Twist":** To properly model the Klein bottle, the distance `d((θ, φ), (θ', φ'))` is `min(d_torus((θ, φ), (θ', φ')), d_torus((θ, φ), (θ'+π, -φ')))`. This can be incorporated into the bias calculation for a true Klein bottle attention.

### Approach C: The Hybrid Gated Approach

This is the most sophisticated method, allowing the model to decide how much to weigh geometric vs. standard semantic attention.

*   **Concept:** Use a learnable, data-dependent gate to mix standard attention scores and the pre-computed topological attention scores.
*   **Implementation:**
    1.  **Compute two attention scores:**
        *   `Scores_Standard = QK^T / sqrt(d_k)`
        *   `Scores_Topological = B_geom` (from Approach B)
    2.  **Gating Mechanism:** For each query `q_i`, compute a gating scalar `γ_i = sigmoid(W_g * q_i)`, where `W_g` is a learnable matrix.
    3.  **Mix Scores:** `FinalScore(i, j) = γ_i * Scores_Standard(i, j) + (1 - γ_i) * Scores_Topological(i, j)`
    4.  Apply softmax as usual: `AttentionWeights = softmax(FinalScores)`

## Phase 4: Analysis and Evaluation

*   **Goal:** Thoroughly evaluate the impact of the modifications.
*   **Metrics:**
    *   Final classification accuracy on CIFAR-10.
    *   Training speed and convergence.
    *   Data efficiency (performance on smaller subsets of the training data).
*   **Analysis:**
    * **ADD**
