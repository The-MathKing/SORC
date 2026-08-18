# Sliced Ollivier-Ricci Curvature (SORC)

Code release for the paper **"Sliced Ollivier-Ricci Curvature for Graph Analysis"**. 

This repository contains the official implementation of the Sliced Ollivier-Ricci Curvature (SORC) approximation scheme, designed to scale the computation of graph curvature for extremely large graphs by exploiting 1D Wasserstein projections along the graph Laplacian eigenvectors.

## Contents
* `sorc.py`: Core implementation of SORC, including the fast sparse spectral projection and 1D Wasserstein distance computations.
* `baselines.py`: Reference implementations of Exact Ollivier-Ricci Curvature (ORC) via linear programming and Augmented Forman-Ricci Curvature (AFRC).
* `experiment_scaling.py`: Script to generate runtime scaling analysis on Stochastic Block Models.
* `experiment_link_prediction.py`: End-to-end link prediction experiment script comparing SORC, AFRC, SVD, and LightGCN on benchmark datasets (Amazon Video Games, MovieLens-1M).
* `datasets.py`: Data loading utilities.

## Requirements
* Python 3.9+
* `numpy`, `scipy`, `networkx`, `scikit-learn`
* `torch`, `torch_geometric` (for LightGCN baseline)

## Usage

You can test the runtime scaling against exact curvature on random graphs:
```bash
python experiment_scaling.py
```

Run the link prediction benchmarks:
```bash
python experiment_link_prediction.py
```
