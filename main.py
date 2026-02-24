# -*- coding: utf-8 -*-
"""
Blended Intensive Program (BIP)
Machine Learning: Mathematical aspects, techniques, and applications
2nd edition

Bank Marketing Dataset - Complete ML Project

Author: Generated for BIP Project
Date: 2024
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import pickle

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("="*80)
print("BIP MACHINE LEARNING PROJECT - BANK MARKETING")
print("="*80)

# ============================================================================
# PART 1: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*80)
print("PART 1: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('../bank+marketing/bank/bank.csv', sep=';')
print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Dataset shape and structure
print("\n2. Dataset Structure:")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Data types
print("\n3. Data Types:")
print(df.dtypes)

# Missing values
print("\n4. Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   No missing values found!")

# Summary statistics
print("\n5. Summary Statistics (Numerical Features):")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numerical_cols].describe())

# Target distribution
print("\n6. Target Variable Distribution:")
target_counts = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100
print(f"   'no':  {target_counts['no']:5d} ({target_pct['no']:.2f}%)")
print(f"   'yes': {target_counts['yes']:5d} ({target_pct['yes']:.2f}%)")

# Save target distribution plot
plt.figure(figsize=(8, 6))
target_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Target Variable Distribution (y)', fontsize=14, fontweight='bold')
plt.xlabel('Subscription (y)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/01_target_distribution.png")

# Numerical feature distributions
print("\n7. Creating histograms for numerical features...")
numerical_features = [col for col in numerical_cols if col != 'y']
n_features = len(numerical_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for i, col in enumerate(numerical_features):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].grid(True, alpha=0.3)

# Hide empty subplots
for i in range(n_features, len(axes)):
    axes[i].axis('off')

plt.suptitle('Numerical Feature Distributions', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/02_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/02_numerical_distributions.png")

# Categorical distributions
print("\n8. Creating bar plots for categorical features...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('y')  # Remove target

n_cat = len(categorical_cols)
n_cols_cat = 3
n_rows_cat = (n_cat + n_cols_cat - 1) // n_cols_cat

fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(18, 5*n_rows_cat))
axes = axes.flatten() if n_cat > 1 else [axes]

for i, col in enumerate(categorical_cols):
    value_counts = df[col].value_counts()
    axes[i].bar(range(len(value_counts)), value_counts.values, color='coral', alpha=0.7)
    axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Count', fontsize=10)
    axes[i].set_xticks(range(len(value_counts)))
    axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for j, v in enumerate(value_counts.values):
        axes[i].text(j, v + max(value_counts.values)*0.01, str(v), 
                    ha='center', va='bottom', fontsize=8)

# Hide empty subplots
for i in range(n_cat, len(axes)):
    axes[i].axis('off')

plt.suptitle('Categorical Feature Distributions', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/03_categorical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/03_categorical_distributions.png")

# Correlation matrix
print("\n9. Creating correlation matrix...")
corr_matrix = df[numerical_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix (Numerical Features)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/04_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/04_correlation_matrix.png")

# Target vs features analysis
print("\n10. Analyzing target variable vs features...")

# Target vs numerical features (box plots)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_features[:6]):  # First 6 numerical features
    df.boxplot(column=col, by='y', ax=axes[i])
    axes[i].set_title(f'{col} by Subscription', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Subscription (y)', fontsize=10)
    axes[i].set_ylabel(col, fontsize=10)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Numerical Features vs Target Variable', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_target_vs_numerical.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/05_target_vs_numerical.png")

# Target vs categorical features
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, col in enumerate(categorical_cols[:9]):  # First 9 categorical features
    crosstab = pd.crosstab(df[col], df['y'], normalize='index') * 100
    crosstab.plot(kind='bar', ax=axes[i], color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[i].set_title(f'{col} vs Subscription', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('Percentage (%)', fontsize=10)
    axes[i].legend(title='Subscription', fontsize=9)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.suptitle('Categorical Features vs Target Variable', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_target_vs_categorical.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/06_target_vs_categorical.png")

print("\n" + "="*80)
print("EDA COMPLETED - All figures saved to figures/ directory")
print("="*80)

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("PART 2: DATA PREPROCESSING")
print("="*80)

# Separate features and target
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})  # Convert to binary

print(f"\n1. Target encoding: 'yes'=1, 'no'=0")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features_list = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n2. Feature types:")
print(f"   Categorical: {len(categorical_features)} features")
print(f"   Numerical: {len(numerical_features_list)} features")

# OneHot Encoding for categorical variables
print("\n3. Applying OneHot Encoding to categorical variables...")
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# Create DataFrame with encoded features
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names, index=X.index)

# Combine numerical and encoded categorical features
X_final = pd.concat([X[numerical_features_list], X_encoded_df], axis=1)

print(f"   Original features: {X.shape[1]}")
print(f"   Final features after encoding: {X_final.shape[1]}")

# Train-test split (80/20)
print("\n4. Splitting data into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Feature scaling
print("\n5. Applying StandardScaler to features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("   ✓ Features scaled (mean=0, std=1)")

# Handle class imbalance
print("\n6. Checking class imbalance...")
class_counts = y_train.value_counts()
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"   Training set class distribution:")
print(f"     Class 0 (no):  {class_counts[0]} ({class_counts[0]/len(y_train)*100:.2f}%)")
print(f"     Class 1 (yes): {class_counts[1]} ({class_counts[1]/len(y_train)*100:.2f}%)")
print(f"   Class weights: {class_weight_dict}")

# Save preprocessing objects
print("\n7. Saving preprocessing objects...")
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump({'encoder': encoder, 'scaler': scaler}, f)
print("   ✓ Saved: preprocessor.pkl")

print("\n" + "="*80)
print("DATA PREPROCESSING COMPLETED")
print("="*80)

# ============================================================================
# PART 3: MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n" + "="*80)
print("PART 3: MODEL TRAINING AND EVALUATION")
print("="*80)

# Function to evaluate model
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion_Matrix': cm
    }

# Store results
results = []

# ----------------------------------------------------------------------------
# Model 1: Logistic Regression
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("-"*80)

# Default parameters
print("\n1. Training with default parameters...")
lr_default = LogisticRegression(random_state=42, max_iter=1000)
lr_default.fit(X_train_scaled, y_train)
y_pred_lr_default = lr_default.predict(X_test_scaled)

results_lr_default = evaluate_model(y_test, y_pred_lr_default, "Logistic Regression (Default)")
results.append(results_lr_default)

print(f"   Accuracy:  {results_lr_default['Accuracy']:.4f}")
print(f"   Precision: {results_lr_default['Precision']:.4f}")
print(f"   Recall:    {results_lr_default['Recall']:.4f}")
print(f"   F1-Score:  {results_lr_default['F1-Score']:.4f}")

# Tuned parameters
print("\n2. Training with tuned parameters (class_weight='balanced', C=0.1)...")
lr_tuned = LogisticRegression(
    random_state=42, 
    max_iter=1000,
    class_weight='balanced',
    C=0.1,
    solver='liblinear'
)
lr_tuned.fit(X_train_scaled, y_train)
y_pred_lr_tuned = lr_tuned.predict(X_test_scaled)

results_lr_tuned = evaluate_model(y_test, y_pred_lr_tuned, "Logistic Regression (Tuned)")
results.append(results_lr_tuned)

print(f"   Accuracy:  {results_lr_tuned['Accuracy']:.4f}")
print(f"   Precision: {results_lr_tuned['Precision']:.4f}")
print(f"   Recall:    {results_lr_tuned['Recall']:.4f}")
print(f"   F1-Score:  {results_lr_tuned['F1-Score']:.4f}")

# Confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(results_lr_default['Confusion_Matrix'], annot=True, fmt='d', 
            cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Logistic Regression (Default)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

sns.heatmap(results_lr_tuned['Confusion_Matrix'], annot=True, fmt='d', 
            cmap='Greens', ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title('Logistic Regression (Tuned)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.suptitle('Logistic Regression - Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_lr_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/07_lr_confusion_matrices.png")

# ----------------------------------------------------------------------------
# Model 2: Decision Tree
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("MODEL 2: DECISION TREE")
print("-"*80)

# Default parameters
print("\n1. Training with default parameters...")
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)
y_pred_dt_default = dt_default.predict(X_test)

results_dt_default = evaluate_model(y_test, y_pred_dt_default, "Decision Tree (Default)")
results.append(results_dt_default)

print(f"   Accuracy:  {results_dt_default['Accuracy']:.4f}")
print(f"   Precision: {results_dt_default['Precision']:.4f}")
print(f"   Recall:    {results_dt_default['Recall']:.4f}")
print(f"   F1-Score:  {results_dt_default['F1-Score']:.4f}")

# Tuned parameters
print("\n2. Training with tuned parameters (max_depth=10, min_samples_split=20, class_weight='balanced')...")
dt_tuned = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced'
)
dt_tuned.fit(X_train, y_train)
y_pred_dt_tuned = dt_tuned.predict(X_test)

results_dt_tuned = evaluate_model(y_test, y_pred_dt_tuned, "Decision Tree (Tuned)")
results.append(results_dt_tuned)

print(f"   Accuracy:  {results_dt_tuned['Accuracy']:.4f}")
print(f"   Precision: {results_dt_tuned['Precision']:.4f}")
print(f"   Recall:    {results_dt_tuned['Recall']:.4f}")
print(f"   F1-Score:  {results_dt_tuned['F1-Score']:.4f}")

# Confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(results_dt_default['Confusion_Matrix'], annot=True, fmt='d', 
            cmap='Oranges', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Decision Tree (Default)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

sns.heatmap(results_dt_tuned['Confusion_Matrix'], annot=True, fmt='d', 
            cmap='Reds', ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title('Decision Tree (Tuned)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.suptitle('Decision Tree - Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_dt_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/08_dt_confusion_matrices.png")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_tuned.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Decision Tree - Top 15 Feature Importances (Tuned)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/09_dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/09_dt_feature_importance.png")

# ----------------------------------------------------------------------------
# Model 3: Random Forest
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("MODEL 3: RANDOM FOREST")
print("-"*80)

# Default parameters
print("\n1. Training with default parameters...")
rf_default = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_default.fit(X_train, y_train)
y_pred_rf_default = rf_default.predict(X_test)

results_rf_default = evaluate_model(y_test, y_pred_rf_default, "Random Forest (Default)")
results.append(results_rf_default)

print(f"   Accuracy:  {results_rf_default['Accuracy']:.4f}")
print(f"   Precision: {results_rf_default['Precision']:.4f}")
print(f"   Recall:    {results_rf_default['Recall']:.4f}")
print(f"   F1-Score:  {results_rf_default['F1-Score']:.4f}")

# Tuned parameters
print("\n2. Training with tuned parameters (n_estimators=200, max_depth=15, class_weight='balanced')...")
rf_tuned = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1
)
rf_tuned.fit(X_train, y_train)
y_pred_rf_tuned = rf_tuned.predict(X_test)

results_rf_tuned = evaluate_model(y_test, y_pred_rf_tuned, "Random Forest (Tuned)")
results.append(results_rf_tuned)

print(f"   Accuracy:  {results_rf_tuned['Accuracy']:.4f}")
print(f"   Precision: {results_rf_tuned['Precision']:.4f}")
print(f"   Recall:    {results_rf_tuned['Recall']:.4f}")
print(f"   F1-Score:  {results_rf_tuned['F1-Score']:.4f}")

# Confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(results_rf_default['Confusion_Matrix'], annot=True, fmt='d', 
            cmap='Purples', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Random Forest (Default)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

sns.heatmap(results_rf_tuned['Confusion_Matrix'], annot=True, fmt='d', 
            cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title('Random Forest (Tuned)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.suptitle('Random Forest - Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_rf_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/10_rf_confusion_matrices.png")

# Feature importance
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_tuned.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance_rf)), feature_importance_rf['importance'], color='coral')
plt.yticks(range(len(feature_importance_rf)), feature_importance_rf['feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest - Top 15 Feature Importances (Tuned)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/11_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/11_rf_feature_importance.png")

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

# Create comparison DataFrame
comparison_df = pd.DataFrame([
    {
        'Model': r['Model'],
        'Accuracy': f"{r['Accuracy']:.4f}",
        'Precision': f"{r['Precision']:.4f}",
        'Recall': f"{r['Recall']:.4f}",
        'F1-Score': f"{r['F1-Score']:.4f}"
    }
    for r in results
])

print("\n" + comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv('model_comparison.csv', index=False)
print("\n   ✓ Saved: model_comparison.csv")

# Visual comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    values = [float(r[metric]) for r in results]
    models = [r['Model'] for r in results]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'wheat', 'plum', 'lightblue']
    
    bars = ax.bar(range(len(models)), values, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for j, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/12_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved: figures/12_model_comparison.png")

# Best model identification
best_f1 = max(results, key=lambda x: x['F1-Score'])
best_accuracy = max(results, key=lambda x: x['Accuracy'])

print(f"\n   Best F1-Score: {best_f1['Model']} (F1={best_f1['F1-Score']:.4f})")
print(f"   Best Accuracy: {best_accuracy['Model']} (Accuracy={best_accuracy['Accuracy']:.4f})")

# Save models
print("\n8. Saving trained models...")
with open('models.pkl', 'wb') as f:
    pickle.dump({
        'lr_default': lr_default,
        'lr_tuned': lr_tuned,
        'dt_default': dt_default,
        'dt_tuned': dt_tuned,
        'rf_default': rf_default,
        'rf_tuned': rf_tuned
    }, f)
print("   ✓ Saved: models.pkl")

print("\n" + "="*80)
print("MODEL TRAINING AND EVALUATION COMPLETED")
print("="*80)

# ============================================================================
# SUMMARY STATISTICS FOR REPORT
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY STATISTICS FOR REPORT")
print("="*80)

# Create summary statistics file
summary_stats = {
    'Dataset': {
        'Total Samples': len(df),
        'Training Samples': len(X_train),
        'Test Samples': len(X_test),
        'Features': X.shape[1],
        'Features after encoding': X_final.shape[1],
        'Target Distribution (no/yes)': f"{target_counts['no']}/{target_counts['yes']}"
    },
    'Models': comparison_df.to_dict('records')
}

import json
with open('summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("   ✓ Saved: summary_stats.json")
print("\n" + "="*80)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nNext steps:")
print("1. Review all generated figures in figures/ directory")
print("2. Check model_comparison.csv for detailed results")
print("3. Generate PDF report using the report generator script")
print("="*80)
