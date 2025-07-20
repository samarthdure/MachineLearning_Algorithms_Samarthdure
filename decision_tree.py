"""
Decision Tree Implementation
Dataset: Titanic Survival Prediction
Author: [Your Name]
Description: This script demonstrates decision tree classification to predict 
passenger survival on the Titanic based on various features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_titanic_data():
    """
    Create a sample Titanic dataset for demonstration
    In real scenarios, you would load from titanic.csv
    """
    np.random.seed(42)
    n_samples = 800
    
    # Generate synthetic Titanic-like data
    data = {
        'Age': np.random.normal(30, 12, n_samples).clip(0, 80),
        'Fare': np.random.lognormal(3, 1, n_samples).clip(0, 500),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'SibSp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
        'Parch': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02])
    }
    
    df = pd.DataFrame(data)
    
    # Create survival based on realistic patterns
    survival_prob = 0.3  # Base survival rate
    survival_prob += (df['Sex'] == 'female') * 0.4  # Women more likely to survive
    survival_prob += (df['Pclass'] == 1) * 0.2  # First class more likely
    survival_prob -= (df['Pclass'] == 3) * 0.15  # Third class less likely
    survival_prob += (df['Age'] < 16) * 0.2  # Children more likely
    survival_prob += (df['Fare'] > 50) * 0.1  # Higher fare more likely
    
    df['Survived'] = (np.random.random(n_samples) < survival_prob).astype(int)
    
    return df

def main():
    print("=" * 50)
    print("DECISION TREE - TITANIC SURVIVAL PREDICTION")
    print("=" * 50)
    
    # Step 1: Load/Create the dataset
    print("\n1. Loading Titanic Dataset...")
    df = create_sample_titanic_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Survival rate: {df['Survived'].mean():.3f}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    # Step 2: Data preprocessing
    print("\n2. Preprocessing the data...")
    
    # Handle categorical variables
    le_sex = LabelEncoder()
    df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
    
    # Select features for the model
    feature_columns = ['Age', 'Fare', 'Pclass', 'Sex_encoded', 'SibSp', 'Parch']
    X = df[feature_columns]
    y = df['Survived']
    
    print(f"Features selected: {feature_columns}")
    print(f"Target: Survived (0=No, 1=Yes)")
    
    # Check for any missing values
    print(f"\nMissing values: {X.isnull().sum().sum()}")
    
    # Step 3: Split the data
    print("\n3. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Create and train the Decision Tree
    print("\n4. Training Decision Tree model...")
    
    # Initialize Decision Tree with reasonable parameters
    dt_model = DecisionTreeClassifier(
        criterion='gini',  # Can also use 'entropy'
        max_depth=5,       # Prevent overfitting
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    # Train the model
    dt_model.fit(X_train, y_train)
    print("Decision Tree training completed!")
    
    # Step 5: Make predictions
    print("\n5. Making predictions...")
    y_pred = dt_model.predict(X_test)
    y_pred_proba = dt_model.predict_proba(X_test)
    
    # Step 6: Evaluate the model
    print("\n6. Model Evaluation:")
    print("-" * 30)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Did not survive', 'Survived']))
    
    # Step 7: Confusion Matrix
    print("\n7. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Did not survive', 'Survived'],
                yticklabels=['Did not survive', 'Survived'])
    plt.title('Confusion Matrix - Decision Tree')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Step 8: Feature importance
    print("\n8. Feature Importance:")
    print("-" * 30)
    
    feature_importance = dt_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(importance_df)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance - Decision Tree')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Step 9: Visualize the Decision Tree (simplified)
    print("\n9. Decision Tree Structure (Text representation):")
    print("-" * 50)
    
    # Show tree rules in text format (first few levels only)
    tree_rules = export_text(dt_model, feature_names=feature_columns, max_depth=3)
    print(tree_rules)
    
    # Plot the decision tree (simplified version)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=feature_columns, 
              class_names=['Did not survive', 'Survived'],
              filled=True, rounded=True, max_depth=3, fontsize=10)
    plt.title('Decision Tree Visualization (Max Depth = 3)')
    plt.show()
    
    # Step 10: Sample predictions
    print("\n10. Sample Predictions:")
    print("-" * 40)
    
    for i in range(min(8, len(X_test))):
        features = X_test.iloc[i]
        actual = 'Survived' if y_test.iloc[i] == 1 else 'Did not survive'
        predicted = 'Survived' if y_pred[i] == 1 else 'Did not survive'
        confidence = max(y_pred_proba[i]) * 100
        
        print(f"Passenger {i+1}:")
        print(f"  Age: {features['Age']:.1f}, Fare: ${features['Fare']:.2f}")
        print(f"  Class: {features['Pclass']}, Sex: {'Female' if features['Sex_encoded']==1 else 'Male'}")
        print(f"  Siblings/Spouses: {features['SibSp']}, Parents/Children: {features['Parch']}")
        print(f"  Actual: {actual}")
        print(f"  Predicted: {predicted}")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Correct: {'✓' if y_test.iloc[i] == y_pred[i] else '✗'}")
        print()
    
    print("=" * 50)
    print("DECISION TREE ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
