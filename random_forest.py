"""
Random Forest Implementation
Dataset: Wine Classification
Author: Samarth Dure
Description: This script demonstrates Random Forest classification for 
wine type recognition using the sklearn wine dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 50)
    print("RANDOM FOREST - WINE CLASSIFICATION")
    print("=" * 50)
    
    # Step 1: Load the Wine dataset
    print("\n1. Loading Wine Dataset...")
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(wine.target_names)}")
    print(f"Classes: {list(wine.target_names)}")
    print(f"Features: {len(wine.feature_names)}")
    
    # Convert to DataFrame for better analysis
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    df = pd.DataFrame(X, columns=feature_names)
    df['wine_class'] = y
    df['wine_name'] = [target_names[i] for i in y]
    
    print(f"\nClass distribution:")
    class_counts = df['wine_name'].value_counts()
    for wine_type, count in class_counts.items():
        print(f"  {wine_type}: {count} samples ({count/len(df)*100:.1f}%)")
    
    print(f"\nFeature names:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    # Step 2: Data exploration
    print("\n2. Data Exploration:")
    print("-" * 25)
    
    print(f"Dataset statistics:")
    print(df[feature_names].describe().round(2))
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color=['red', 'white', 'rosé'])
    plt.title('Wine Class Distribution')
    plt.xlabel('Wine Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap of top features
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[feature_names].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Step 3: Data preprocessing (optional for Random Forest)
    print("\n3. Data Preprocessing:")
    print("-" * 25)
    
    # Random Forest doesn't require scaling, but we'll show both approaches
    print("Random Forest doesn't require feature scaling, but we'll test both:")
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Optional: Create scaled version for comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Hyperparameter tuning
    print("\n4. Finding optimal hyperparameters...")
    
    # Test different parameters
    param_combinations = [
        {'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2},
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
    ]
    
    best_score = 0
    best_params = None
    results = []
    
    for params in param_combinations:
        rf = RandomForestClassifier(random_state=42, **params)
        scores = cross_val_score(rf, X_train, y_train, cv=5)
        mean_score = scores.mean()
        std_score = scores.std()
        
        results.append({
            'params': params,
            'cv_score': mean_score,
            'cv_std': std_score
        })
        
        print(f"Params: {params}")
        print(f"  CV Score: {mean_score:.4f} (+/- {std_score*2:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    # Step 5: Train the final Random Forest model
    print("\n5. Training Random Forest model...")
    
    rf_model = RandomForestClassifier(
        random_state=42,
        **best_params,
        oob_score=True  # Out-of-bag score for additional validation
    )
    
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed!")
    print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")
    
    # Step 6: Make predictions
    print("\n6. Making predictions...")
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Step 7: Evaluate the model
    print("\n7. Model Evaluation:")
    print("-" * 25)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Step 8: Confusion Matrix
    print("\n8. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted Wine Type')
    plt.ylabel('Actual Wine Type')
    plt.tight_layout()
    plt.show()
    
    # Step 9: Feature Importance Analysis
    print("\n9. Feature Importance Analysis:")
    print("-" * 35)
    
    feature_importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(importance_df.head(10).round(4))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('Top 15 Feature Importance - Random Forest')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Cumulative importance
    cumulative_importance = np.cumsum(importance_df['importance'].values)
    n_features_90 = np.where(cumulative_importance >= 0.90)[0][0] + 1
    n_features_95 = np.where(cumulative_importance >= 0.95)[0][0] + 1
    
    print(f"\nCumulative Feature Importance:")
    print(f"  Top {n_features_90} features explain 90% of importance")
    print(f"  Top {n_features_95} features explain 95% of importance")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_importance)+1), cumulative_importance, 'bo-')
    plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Step 10: Individual tree analysis
    print("\n10. Forest Composition Analysis:")
    print("-" * 35)
    
    n_trees = rf_model.n_estimators
    tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
    tree_nodes = [tree.tree_.node_count for tree in rf_model.estimators_]
    
    print(f"Forest composition:")
    print(f"  Number of trees: {n_trees}")
    print(f"  Average tree depth: {np.mean(tree_depths):.2f} (+/- {np.std(tree_depths):.2f})")
    print(f"  Average nodes per tree: {np.mean(tree_nodes):.1f} (+/- {np.std(tree_nodes):.1f})")
    print(f"  Min tree depth: {min(tree_depths)}")
    print(f"  Max tree depth: {max(tree_depths)}")
    
    # Visualize tree statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(tree_depths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Tree Depth')
    ax1.set_ylabel('Number of Trees')
    ax1.set_title('Distribution of Tree Depths')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.hist(tree_nodes, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Number of Trees')
    ax2.set_title('Distribution of Tree Sizes')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Step 11: Prediction confidence analysis
    print("\n11. Prediction Confidence Analysis:")
    print("-" * 40)
    
    # Calculate prediction confidence (max probability)
    prediction_confidence = np.max(y_pred_proba, axis=1)
    
    print(f"Prediction confidence statistics:")
    print(f"  Mean confidence: {np.mean(prediction_confidence):.4f}")
    print(f"  Min confidence: {np.min(prediction_confidence):.4f}")
    print(f"  Max confidence: {np.max(prediction_confidence):.4f}")
    print(f"  Std deviation: {np.std(prediction_confidence):.4f}")
    
    # Confidence by correctness
    correct_predictions = (y_pred == y_test)
    correct_confidence = prediction_confidence[correct_predictions]
    incorrect_confidence = prediction_confidence[~correct_predictions]
    
    if len(incorrect_confidence) > 0:
        print(f"\nConfidence comparison:")
        print(f"  Correct predictions: {np.mean(correct_confidence):.4f} confidence")
        print(f"  Incorrect predictions: {np.mean(incorrect_confidence):.4f} confidence")
    
    # Visualize confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_confidence, bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Number of Predictions')
    plt.title('Distribution of Prediction Confidence')
    plt.axvline(np.mean(prediction_confidence), color='red', linestyle='--', 
                label=f'Mean: {np.mean(prediction_confidence):.3f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # Step 12: Sample predictions
    print("\n12. Sample Predictions with Details:")
    print("-" * 42)
    
    for i in range(min(8, len(X_test))):
        actual = target_names[y_test[i]]
        predicted = target_names[y_pred[i]]
        confidence = max(y_pred_proba[i]) * 100
        
        print(f"Wine Sample {i+1}:")
        print(f"  Actual type: {actual}")
        print(f"  Predicted type: {predicted}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Show all class probabilities
        print(f"  Class probabilities:")
        for j, wine_type in enumerate(target_names):
            prob = y_pred_proba[i][j] * 100
            print(f"    {wine_type}: {prob:.1f}%")
        
        print(f"  Correct: {'✓' if y_test[i] == y_pred[i] else '✗'}")
        print()
    
    # Step 13: Comparison with scaled data (bonus)
    print("\n13. Comparison: Scaled vs Unscaled Data:")
    print("-" * 45)
    
    rf_scaled = RandomForestClassifier(**best_params, random_state=42)
    rf_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = rf_scaled.predict(X_test_scaled)
    accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
    
    print(f"Unscaled data accuracy: {accuracy:.4f}")
    print(f"Scaled data accuracy: {accuracy_scaled:.4f}")
    print(f"Difference: {abs(accuracy - accuracy_scaled):.4f}")
    print("Note: Random Forest is generally insensitive to feature scaling")
    
    # Step 14: Model summary
    print("\n14. Model Summary:")
    print("-" * 20)
    print(f"Algorithm: Random Forest")
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Max depth: {rf_model.max_depth}")
    print(f"Min samples split: {rf_model.min_samples_split}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(target_names)}")
    print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")
    
    print("\n" + "=" * 50)
    print("RANDOM FOREST ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
