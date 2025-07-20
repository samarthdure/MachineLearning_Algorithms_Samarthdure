"""
Logistic Regression Implementation
Dataset: Iris Classification
Author: Samarth Dure
Description: This script demonstrates logistic regression for multi-class classification
using the famous Iris dataset to predict flower species.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 50)
    print("LOGISTIC REGRESSION - IRIS CLASSIFICATION")
    print("=" * 50)
    
    # Step 1: Load the Iris dataset
    print("\n1. Loading Iris Dataset...")
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)
    
    # Convert to DataFrame for better visualization
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(feature_names)}")
    print(f"Target classes: {list(target_names)}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Step 2: Split the data into training and testing sets
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Step 3: Create and train the logistic regression model
    print("\n3. Training Logistic Regression model...")
    
    # Initialize the model
    logistic_model = LogisticRegression(
        random_state=42,
        max_iter=1000,  # Increase iterations for convergence
        multi_class='ovr'  # One-vs-Rest for multi-class
    )
    
    # Train the model
    logistic_model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    y_pred = logistic_model.predict(X_test)
    y_pred_proba = logistic_model.predict_proba(X_test)
    
    # Step 5: Evaluate the model
    print("\n5. Model Evaluation:")
    print("-" * 30)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Step 6: Display confusion matrix
    print("\n6. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Step 7: Show some individual predictions
    print("\n7. Sample Predictions:")
    print("-" * 40)
    for i in range(min(10, len(X_test))):
        actual_class = target_names[y_test[i]]
        predicted_class = target_names[y_pred[i]]
        confidence = max(y_pred_proba[i]) * 100
        
        print(f"Sample {i+1}:")
        print(f"  Features: {X_test[i]}")
        print(f"  Actual: {actual_class}")
        print(f"  Predicted: {predicted_class}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Correct: {'✓' if y_test[i] == y_pred[i] else '✗'}")
        print()
    
    # Step 8: Feature importance (coefficients)
    print("\n8. Feature Importance (Coefficients):")
    print("-" * 40)
    
    # For multi-class, we get coefficients for each class
    for i, class_name in enumerate(target_names):
        print(f"\nClass: {class_name}")
        for j, feature_name in enumerate(feature_names):
            coef = logistic_model.coef_[i][j]
            print(f"  {feature_name}: {coef:.4f}")
    
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
