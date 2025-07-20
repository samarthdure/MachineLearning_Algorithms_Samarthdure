"""
Support Vector Machine (SVM) Implementation
Dataset: Breast Cancer Classification (Malignant vs Benign)
Author: Samarth Dure
Description: This script demonstrates SVM classification for breast cancer 
diagnosis using the Wisconsin Breast Cancer dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 55)
    print("SUPPORT VECTOR MACHINE - BREAST CANCER CLASSIFICATION")
    print("=" * 55)
    
    # Step 1: Load the Breast Cancer dataset
    print("\n1. Loading Breast Cancer Dataset...")
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Classes: {cancer.target_names}")  # ['malignant', 'benign']
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"  {cancer.target_names[class_idx]}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Convert to DataFrame for better understanding
    feature_names = cancer.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"\nFirst few feature names:")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i+1}. {name}")
    print("  ... and 20 more features")
    
    print(f"\nDataset statistics:")
    print(df.describe().loc[['mean', 'std', 'min', 'max']].round(3))
    
    # Step 2: Data preprocessing
    print("\n2. Preprocessing the data...")
    
    # Feature scaling is crucial for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Features scaled using StandardScaler")
    print(f"Original feature ranges vary widely:")
    print(f"  Min values: {X.min(axis=0)[:5].round(3)} ...")
    print(f"  Max values: {X.max(axis=0)[:5].round(3)} ...")
    print(f"After scaling - all features have mean ≈ 0, std ≈ 1")
    
    # Step 3: Split the data
    print("\n3. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Hyperparameter tuning
    print("\n4. Finding optimal hyperparameters...")
    print("This may take a moment...")
    
    # Define parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],           # Regularization parameter
        'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient
        'kernel': ['rbf', 'linear']        # Kernel type
    }
    
    # Perform grid search with cross-validation
    svm_grid = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm_grid, param_grid, cv=5, 
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Step 5: Train the final SVM model
    print("\n5. Training SVM model with best parameters...")
    
    best_svm = grid_search.best_estimator_
    print(f"Final model: {best_svm}")
    
    # Step 6: Make predictions
    print("\n6. Making predictions...")
    y_pred = best_svm.predict(X_test)
    
    # For probability predictions, we need probability=True
    svm_proba = SVC(**grid_search.best_params_, probability=True, random_state=42)
    svm_proba.fit(X_train, y_train)
    y_pred_proba = svm_proba.predict_proba(X_test)
    
    # Step 7: Evaluate the model
    print("\n7. Model Evaluation:")
    print("-" * 30)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=cancer.target_names))
    
    # Step 8: Confusion Matrix
    print("\n8. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cancer.target_names,
                yticklabels=cancer.target_names)
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    precision = tp / (tp + fp)
    
    print(f"\nAdditional Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    
    # Step 9: Feature analysis with Linear SVM
    print("\n9. Feature Importance Analysis:")
    print("-" * 35)
    
    if grid_search.best_params_['kernel'] == 'linear':
        # For linear SVM, we can analyze coefficients
        coefficients = best_svm.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))
        
        # Visualize top features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
        
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('SVM Coefficient')
        plt.title('Top 15 Feature Coefficients (Linear SVM)')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    else:
        print(f"Feature importance analysis not available for {grid_search.best_params_['kernel']} kernel")
        print("Linear kernel required for coefficient interpretation")
    
    # Step 10: Decision boundary visualization (2D projection)
    print("\n10. Decision Boundary Visualization:")
    print("-" * 40)
    
    # Use the two most important features for 2D visualization
    if grid_search.best_params_['kernel'] == 'linear':
        top_2_features = feature_importance.head(2)['feature'].values
        feature_indices = [list(feature_names).index(f) for f in top_2_features]
    else:
        # Use first two features if not linear
        feature_indices = [0, 1]
        top_2_features = feature_names[feature_indices]
    
    print(f"Visualizing decision boundary using features:")
    print(f"  X-axis: {top_2_features[0]}")
    print(f"  Y-axis: {top_2_features[1]}")
    
    # Extract 2D data
    X_2d = X_scaled[:, feature_indices]
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train 2D SVM
    svm_2d = SVC(**grid_search.best_params_, random_state=42)
    svm_2d.fit(X_train_2d, y_train_2d)
    
    # Create mesh for decision boundary
    h = 0.1
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot training points
    scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d, 
                         cmap='RdYlBu', alpha=0.7, s=50)
    plt.colorbar(scatter, ticks=[0, 1], label='Class')
    
    # Plot support vectors if available
    if hasattr(svm_2d, 'support_vectors_'):
        plt.scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1], 
                   s=100, facecolors='none', edgecolors='black', linewidth=2,
                   label='Support Vectors')
        plt.legend()
    
    plt.xlabel(top_2_features[0])
    plt.ylabel(top_2_features[1])
    plt.title(f'SVM Decision Boundary ({grid_search.best_params_["kernel"]} kernel)')
    plt.show()
    
    print(f"2D SVM accuracy: {svm_2d.score(X_test_2d, y_test_2d):.4f}")
    if hasattr(svm_2d, 'support_vectors_'):
        print(f"Number of support vectors: {len(svm_2d.support_vectors_)}")
    
    # Step 11: Sample predictions with confidence
    print("\n11. Sample Predictions with Confidence:")
    print("-" * 45)
    
    for i in range(min(10, len(X_test))):
        actual = cancer.target_names[y_test[i]]
        predicted = cancer.target_names[y_pred[i]]
        confidence = max(y_pred_proba[i]) * 100
        
        print(f"Patient {i+1}:")
        print(f"  Actual diagnosis: {actual}")
        print(f"  Predicted diagnosis: {predicted}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Show probability breakdown
        prob_malignant = y_pred_proba[i][0] * 100
        prob_benign = y_pred_proba[i][1] * 100
        print(f"  Probabilities: Malignant={prob_malignant:.1f}%, Benign={prob_benign:.1f}%")
        print(f"  Correct: {'✓' if y_test[i] == y_pred[i] else '✗'}")
        print()
    
    # Step 12: Model summary
    print("\n12. Model Summary:")
    print("-" * 25)
    print(f"Algorithm: Support Vector Machine")
    print(f"Kernel: {grid_search.best_params_['kernel']}")
    print(f"C (Regularization): {grid_search.best_params_['C']}")
    print(f"Gamma: {grid_search.best_params_['gamma']}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if hasattr(best_svm, 'support_'):
        print(f"Support vectors: {len(best_svm.support_)}")
    
    print("\n" + "=" * 55)
    print("SUPPORT VECTOR MACHINE ANALYSIS COMPLETE")
    print("=" * 55)

if __name__ == "__main__":
    main()