# 📉 Dimensionality Reduction Techniques in Machine Learning

Dimensionality reduction simplifies high-dimensional data while preserving its essential structure. It improves model performance, reduces overfitting, and enhances interpretability.

---

## 🔷 Linear Techniques

| Technique | Description | Pros | Cons | ML Tasks | Real-World Use Cases |
|----------|-------------|------|------|----------|-----------------------|
| **PCA (Principal Component Analysis)** | Projects data onto directions of maximum variance using orthogonal linear transformations. | Fast, interpretable, preserves global structure. | Assumes linearity, sensitive to scaling. | Clustering, Regression, Visualization | Image compression, gene expression analysis, exploratory data analysis. |
| **LDA (Linear Discriminant Analysis)** | Maximizes class separability by projecting data onto a lower-dimensional space. | Good for classification tasks, supervised. | Requires labeled data, assumes Gaussian distribution. | Classification | Face recognition, document classification. |
| **SVD (Singular Value Decomposition)** | Matrix factorization technique used to decompose data into singular vectors and values. | Powerful for sparse data, used in latent semantic analysis. | Computationally expensive for large datasets. | Clustering, Feature Extraction | Recommender systems, topic modeling. |
| **Factor Analysis** | Models observed variables as linear combinations of latent factors plus noise. | Captures hidden structure, useful for psychometrics. | Assumes linearity and Gaussian noise. | Regression, Feature Engineering | Market research, behavioral science. |
| **Random Projection** | Projects data using a random matrix while preserving pairwise distances. | Fast, scalable, preserves geometry. | Less interpretable, may lose structure. | Clustering, Classification | Text mining, large-scale clustering. |

---

## 🔶 Non-Linear Techniques

| Technique | Description | Pros | Cons | ML Tasks | Real-World Use Cases |
|----------|-------------|------|------|----------|-----------------------|
| **t-SNE (t-Distributed Stochastic Neighbor Embedding)** | Preserves local structure by modeling pairwise similarities in high and low dimensions. | Excellent for visualization, captures non-linear patterns. | Computationally intensive, poor global structure. | Visualization, Clustering | Visualizing word embeddings, image clustering. |
| **UMAP (Uniform Manifold Approximation and Projection)** | Preserves both local and global structure using manifold learning. | Fast, scalable, better global structure than t-SNE. | Sensitive to hyperparameters. | Visualization, Clustering | Genomics, NLP embeddings, anomaly detection. |
| **Isomap** | Preserves geodesic distances on a manifold using neighborhood graphs. | Captures global non-linear structure. | Sensitive to noise, slow for large datasets. | Clustering, Visualization | Speech recognition, 3D shape analysis. |
| **LLE (Locally Linear Embedding)** | Preserves local neighborhood relationships using linear reconstructions. | Good for unfolding manifolds. | Poor global structure, sensitive to noise. | Visualization | Handwritten digit visualization, motion capture data. |
| **Autoencoders** | Neural networks that learn compressed representations through reconstruction. | Flexible, scalable, can be stacked or variational. | Requires tuning, less interpretable. | Feature Extraction, Anomaly Detection, Classification | Image denoising, anomaly detection, feature extraction. |
| **Kernel PCA** | Extends PCA using kernel methods to capture non-linear structure. | Captures complex relationships, customizable kernels. | Computationally expensive, kernel selection is critical. | Classification, Clustering | Non-linear feature extraction, pattern recognition. |

---

## 🧠 Summary

- **Linear methods** are ideal for structured tabular data and interpretable models.
- **Non-linear methods** excel in unstructured data (images, text, audio) and complex manifolds.
- **ML Task Alignment**:
  - 📊 **Visualization**: t-SNE, UMAP, LLE, PCA
  - 🔍 **Clustering**: PCA, UMAP, t-SNE, Isomap, Random Projection
  - 🧠 **Classification**: LDA, Autoencoders, Kernel PCA
  - 📈 **Regression/Feature Engineering**: PCA, Factor Analysis, Autoencoders

---
## Decision Tree 

Start
 └──► Is your data structured/tabular?
       ├─ Yes → PCA or LDA
       └─ No → Is it image/text/audio?
               ├─ Yes → Autoencoders or UMAP
               └─ No → Use Kernel PCA or t-SNE

 └──► Is interpretability critical?
       ├─ Yes → PCA or Factor Analysis
       └─ No → Autoencoders or Random Projection

 └──► Is your goal visualization or clustering?
       ├─ Visualization → t-SNE or UMAP
       └─ Clustering → PCA or Isomap

 └──► Is your task classification or regression?
       ├─ Classification → LDA or Kernel PCA
       └─ Regression → PCA or Autoencoders

 └──► Is your data high-volume or real-time?
       ├─ Yes → UMAP or Random Projection
       └─ No → Choose based on interpretability vs. accuracy
