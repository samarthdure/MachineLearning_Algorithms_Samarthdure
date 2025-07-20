"""
K-Nearest Neighbors (KNN) Implementation
Dataset: Digits Recognition (0-9)
Author: Samarth Dure
Description: This script demonstrates KNN classification for recognizing 
handwritten digits using the sklearn digits dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 50)
    print("K-NEAREST NEIGHBORS - DIGIT RECOGNITION")
    print("=" * 50)
    
    # Step 1: Load the Digits dataset
    print("\n1. Loading Digits Dataset...")
    digits = load_digits()
    X = digits.data  # 8x8 pixel images flattened to 64 features
    y = digits.target  # Digits 0-9
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    print(f"Each image is 8x8 pixels = {X.shape[1]} features")
    
    # Display some sample digits
    print("\n2. Visualizing Sample Digits:")
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(digits.images[i], cmap='gray')
        axes[row, col].set_title(f'Digit: {digits.target[i]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Step 3: Data preprocessing
    print("\n3. Preprocessing the data...")
    
    # Normalize the data (important for KNN as it uses distance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data normalized using StandardScaler")
    print(f"Original data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    
    # Step 4: Split the data
    print("\n4. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Find optimal K value
    print("\n5. Finding optimal K value...")
    
    # Test different K values
    k_values = range(1, 21)
    cv_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        # Use cross-validation to get more reliable estimate
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())
        if k <= 10 or k % 5 == 0:  # Print every value up to 10, then every 5th
            print(f"K={k}: CV Accuracy = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # Find best K
    best_k = k_values[np.argmax(cv_scores)]
    best_score = max(cv_scores)
    print(f"\nBest K value: {best_k} with CV accuracy: {best_score:.4f}")
    
    # Plot K values vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Best K = {best_k}')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN Performance vs K Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Step 6: Train the final KNN model
    print(f"\n6. Training KNN model with K={best_k}...")
    
    knn_model = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='uniform',  # Can also try 'distance'
        algorithm='auto',   # Automatically choose best algorithm
        metric='euclidean'  # Distance metric
    )
    
    knn_model.fit(X_train, y_train)
    print("KNN model training completed!")
    
    # Step 7: Make predictions
    print("\n7. Making predictions...")
    y_pred = knn_model.predict(X_test)
    y_pred_proba = knn_model.predict_proba(X_test)
    
    # Step 8: Evaluate the model
    print("\n8. Model Evaluation:")
    print("-" * 30)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 9: Confusion Matrix
    print("\n9. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - KNN (K={best_k})')
    plt.xlabel('Predicted Digit')
    plt.ylabel('Actual Digit')
    plt.show()
    
    # Step 10: Analyze per-digit accuracy
    print("\n10. Per-Digit Accuracy Analysis:")
    print("-" * 40)
    
    digit_accuracy = []
    for digit in range(10):
        digit_mask = (y_test == digit)
        if digit_mask.sum() > 0:
            digit_acc = (y_pred[digit_mask] == digit).mean()
            digit_accuracy.append(digit_acc)
            print(f"Digit {digit}: {digit_acc:.4f} ({digit_acc*100:.2f}%) - {digit_mask.sum()} samples")
        else:
            digit_accuracy.append(0)
    
    # Visualize per-digit accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(10), digit_accuracy, color='skyblue', edgecolor='navy')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.title('Per-Digit Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # Step 11: Show misclassified examples
    print("\n11. Analyzing Misclassified Examples:")
    print("-" * 45)
    
    # Find misclassified samples
    misclassified = X_test[y_test != y_pred]
    misclassified_true = y_test[y_test != y_pred]
    misclassified_pred = y_pred[y_test != y_pred]
    
    print(f"Total misclassified: {len(misclassified)}")
    
    if len(misclassified) > 0:
        # Show first few misclassified examples
        n_show = min(8, len(misclassified))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(n_show):
            # Inverse transform to get back original pixel values
            img_data = scaler.inverse_transform(misclassified[i].reshape(1, -1))
            img = img_data.reshape(8, 8)
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {misclassified_true.iloc[i]}, Pred: {misclassified_pred[i]}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_show, 8):
            axes[i].axis('off')
            
        plt.suptitle('Misclassified Examples')
        plt.tight_layout()
        plt.show()
    
    # Step 12: Sample predictions with confidence
    print("\n12. Sample Predictions with Confidence:")
    print("-" * 45)
    
    for i in range(min(10, len(X_test))):
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predicted = y_pred[i]
        confidence = max(y_pred_proba[i]) * 100
        
        print(f"Sample {i+1}:")
        print(f"  Actual digit: {actual}")
        print(f"  Predicted digit: {predicted}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Correct: {'✓' if actual == predicted else '✗'}")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(y_pred_proba[i])[-3:][::-1]
        print(f"  Top 3 predictions:")
        for j, idx in enumerate(top_3_indices):
            prob = y_pred_proba[i][idx] * 100
            print(f"    {j+1}. Digit {idx}: {prob:.2f}%")
        print()
    
    print("=" * 50)
    print("K-NEAREST NEIGHBORS ANALYSIS COMPLETE")
    print(f"Final Model: KNN with K={best_k}, Test Accuracy: {accuracy:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
