# Machine Learning Algorithms Collection

This repository contains implementations of five fundamental machine learning algorithms using Python and scikit-learn. Each script demonstrates a complete machine learning workflow including data loading, preprocessing, training, evaluation, and visualization.

## 📊 Overview

| Algorithm | Dataset | Problem Type | Key Features |
|-----------|---------|--------------|-------------|
| Logistic Regression | Iris | Multi-class Classification | Linear decision boundaries, probability outputs |
| Decision Tree | Titanic | Binary Classification | Interpretable rules, feature importance |
| K-Nearest Neighbors | Digits | Multi-class Classification | Instance-based learning, distance metrics |
| Support Vector Machine | Breast Cancer | Binary Classification | Kernel trick, margin optimization |
| Random Forest | Wine | Multi-class Classification | Ensemble method, feature importance |

## 🔧 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

**Required packages:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning library
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization

## 📁 File Descriptions

### 1. `1_logistic_regression.py`
**Algorithm:** Logistic Regression  
**Dataset:** Iris flower classification  
**Purpose:** Demonstrates linear classification for multi-class problems

**Key Features:**
- Multi-class classification (3 species)
- Feature coefficient analysis
- Confusion matrix visualization
- Prediction confidence analysis

**Sample Output:**
```
LOGISTIC REGRESSION - IRIS CLASSIFICATION
Dataset shape: (150, 5)
Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target classes: ['setosa', 'versicolor', 'virginica']
Test Accuracy: 1.0000 (100.00%)
```

### 2. `2_decision_tree.py`
**Algorithm:** Decision Tree Classifier  
**Dataset:** Titanic passenger survival (synthetic data)  
**Purpose:** Shows interpretable tree-based classification

**Key Features:**
- Synthetic Titanic dataset generation
- Tree visualization and rules
- Feature importance ranking
- Interpretable decision paths

**Sample Output:**
```
DECISION TREE - TITANIC SURVIVAL PREDICTION
Dataset shape: (800, 7)
Survival rate: 0.384
Test Accuracy: 0.8125 (81.25%)
```

### 3. `3_knn.py`
**Algorithm:** K-Nearest Neighbors  
**Dataset:** Handwritten digits (0-9)  
**Purpose:** Demonstrates instance-based learning

**Key Features:**
- Optimal K value selection via cross-validation
- Digit image visualization
- Per-digit accuracy analysis
- Misclassification examples

**Sample Output:**
```
K-NEAREST NEIGHBORS - DIGIT RECOGNITION
Dataset shape: (1797, 64)
Number of classes: 10
Best K value: 3 with CV accuracy: 0.9888
Test Accuracy: 0.9889 (98.89%)
```

### 4. `4_svm.py`
**Algorithm:** Support Vector Machine  
**Dataset:** Breast cancer diagnosis  
**Purpose:** Shows kernel-based classification for medical diagnosis

**Key Features:**
- Hyperparameter tuning with GridSearchCV
- Kernel comparison (RBF vs Linear)
- Decision boundary visualization
- Support vector analysis

**Sample Output:**
```
SUPPORT VECTOR MACHINE - BREAST CANCER CLASSIFICATION
Dataset shape: (569, 30)
Classes: ['malignant' 'benign']
Best parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
Test Accuracy: 0.9737 (97.37%)
```

### 5. `5_random_forest.py`
**Algorithm:** Random Forest Classifier  
**Dataset:** Wine type recognition  
**Purpose:** Demonstrates ensemble learning methods

**Key Features:**
- Forest composition analysis
- Feature importance ranking
- Out-of-bag score validation
- Tree depth and size statistics

**Sample Output:**
```
RANDOM FOREST - WINE