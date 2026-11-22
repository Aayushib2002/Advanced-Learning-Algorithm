# Advanced Learning Algorithms

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189FDD)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)

A collection of Python and TensorFlow implementations covering neural networks from scratch, multi-class deep learning pipelines, machine learning diagnostics, and tree-based ensemble models.

---

## Core Topics & Implementations

### 1. Neural Networks & Forward Propagation
- Vectorized dense layer inference built from scratch using NumPy.
- Sequential neural network construction and binary classification in TensorFlow/Keras.
- Matrix multiplication optimizations for batch forward passes.

### 2. Neural Network Training & Multi-Class Classification
- Activation functions: Sigmoid, ReLU, Linear, and Softmax.
- Numerically stable multi-class classification using `from_logits=True` with Sparse Categorical Cross-Entropy.
- Multi-label classification architectures, Adam optimization, and custom layer weight inspection.
- Backpropagation and computational graph derivatives.

### 3. Model Diagnostics & ML Engineering Best Practices
- Model selection and evaluation across train, cross-validation, and test splits.
- Bias vs. variance diagnostics, regularization tuning ($\lambda$), and learning curve analysis.
- Iterative error analysis, data augmentation, and transfer learning workflows.
- Skewed dataset evaluation using Precision, Recall, and $F_1$ score.

### 4. Decision Trees & Ensemble Methods
- From-scratch decision tree construction using entropy and information gain.
- Handling categorical features (one-hot encoding) and continuous feature thresholds.
- Tree ensembles: Bagging, Random Forests, and Gradient Boosted Trees (`XGBoost`) with hyperparameter tuning.

---

## Repository Structure

```text
.
├── 01_neural_networks/        # NumPy forward propagation & Keras binary classifiers
├── 02_nn_training/            # Multi-class classification, Softmax, ReLU & Adam
├── 03_ml_diagnostics/         # Bias/variance analysis, regularization & error metrics
├── 04_tree_ensembles/         # Decision trees from scratch, Random Forest & XGBoost
└── utils/                     # Helper scripts, plotting functions & data loaders
```

---

## Tech Stack

- **Core:** Python 3, NumPy, Pandas
- **Deep Learning:** TensorFlow / Keras
- **Classical ML & Ensembles:** Scikit-learn, XGBoost
- **Visualization & Environment:** Matplotlib, Jupyter Notebook

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow xgboost jupyter
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
