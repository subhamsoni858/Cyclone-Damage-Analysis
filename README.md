# 🌀 Cyclone Damage Assessment using Percolation-Aware GNNs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red)
![Status](https://img.shields.io/badge/Status-Research_Complete-success)

**A Graph Neural Network (GNN) approach to mapping mangrove ecosystem resilience.**

## 🚀 Project Overview

Traditional remote sensing models (like Random Forest or CNNs) treat satellite images as isolated pixels, often failing to capture the structural fragmentation caused by cyclones.

This project introduces a **Percolation-Aware GNN** that models the forest as a connected graph. By analyzing the connectivity and topology of the ecosystem, we can detect "tipping points" where minor canopy thinning transitions into severe structural collapse.

## 📊 Key Results

We moved beyond standard metrics to achieve physically realistic damage maps:

| Metric                | Performance                                  |
| :-------------------- | :------------------------------------------- |
| **Overall Accuracy**  | **~80%** (5-Fold Spatial Block CV)           |
| **F1-Macro Score**    | **0.74** (High sensitivity to severe damage) |
| **Kappa Coefficient** | **0.61** (Substantial Agreement)             |

> **Novelty:** Unlike baseline pixel-based methods that struggle with class imbalance, our **Residual GraphSAGE** model (trained with Weighted Random Sampling) successfully identifies rare, severe damage corridors with high precision.

## 🖼️ Visuals

![Final Damage Map](final_damage_map.jpg)
_Figure 1: Final classification map. Red zones indicate severe structural fragmentation (loss of percolation), while Yellow zones indicate thinned but connected canopy._

## 🛠️ Methodology

1.  **Graph Construction:** Segments Sentinel-2 imagery into 10,000 superpixels and builds a spatial graph ($k=6$ neighbors).
2.  **Percolation Features:** Extracts topological metrics (Node Degree, Component Size) to measure forest connectivity.
3.  **Model Architecture:** Deep **Residual GraphSAGE** (3-Layer) to capture diffusive damage patterns.
4.  **Rigorous Validation:** Uses **5-Fold Spatial Block Cross-Validation** to prevent spatial data leakage.

## 💻 Tech Stack

- **Core:** Python, PyTorch Geometric, NetworkX
- **Geo-Spatial:** Rasterio, GeoPandas, Scikit-Image
- **Machine Learning:** Scikit-Learn (GPR for Uncertainty estimation)

## 📂 Project Structure

```text
├── Cyclone_GNN_Final_Model.ipynb   # The complete training & validation pipeline
├── region_folder/                  # (GitIgnored) Stores raw Sentinel-2 .tif files
├── final_damage_map.jpg            # Output visualization
└── README.md                       # Project documentation
```
