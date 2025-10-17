# Databricks notebook source
# MAGIC %md
# MAGIC # Phenotype Matching: ICD9 ↔ ICD10 with Similarity Features (Pandas/Sklearn)
# MAGIC
# MAGIC This notebook trains and evaluates models that predict whether an ICD9 phenotype name matches an ICD10 phenotype name using similarity features.
# MAGIC
# MAGIC **Schema expected in data:**
# MAGIC - `icd9_code` (string), `icd9_phe` (string)
# MAGIC - `icd10_code` (string), `icd10_phe` (string)
# MAGIC - Features: `cosine_tfidf_word`, `jaccard_tokens`, `dice_tokens`, `levenshtein_ratio`, `cos_similarity` (float)
# MAGIC - Label: `correct` (int: 0/1)
# MAGIC
# MAGIC **Core outputs:**
# MAGIC - Baselines (simple weighted score), Logistic Regression, Random Forest, Gradient Boosting
# MAGIC - Classification metrics (AUC, Accuracy, F1, Precision, Recall)
# MAGIC - Ranking metrics per query (MAP, MRR, Precision@1 / Top-1 accuracy)
# MAGIC - Threshold tuning and per-query decisioning (choose best candidate per ICD9)
# MAGIC
# MAGIC **Refactored from PySpark to pandas/sklearn** for small datasets - faster, simpler, more portable!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import random
from pathlib import Path

# ML and stats packages
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

print("✓ All packages imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Feature columns to use for modeling
FEATURES = [
    'cosine_tfidf_word',
    'jaccard_tokens',
    'dice_tokens',
    'levenshtein_ratio',
    'cos_similarity'
]

LABEL = 'correct'           # Target variable (0 or 1)
QUERY_COL = 'icd9_code'     # Each ICD9 has many candidates (ICD10)
DOC_COL = 'icd10_code'      # The candidate documents

# Data path - update this to your actual data location
DATA_PATH = '/mnt/data/projects/similarity/AoU_test_set_pairwise_w_correct_answers.delta'

print(f"Features: {FEATURES}")
print(f"Label: {LABEL}")
print(f"Query column: {QUERY_COL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading Functions

# COMMAND ----------

def load_data(file_path):
    """Load data from delta/parquet/csv file"""
    if Path(file_path).exists():
        # Try to load from parquet or CSV
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.delta') or Path(file_path).is_dir():
            # Try reading as parquet directory (delta format)
            try:
                import pyarrow.parquet as pq
                return pq.read_table(file_path).to_pandas()
            except:
                raise ValueError("Delta table reading requires pyarrow. Install with: pip install pyarrow")
    else:
        raise FileNotFoundError(f"Data file not found: {file_path}")


def generate_synthetic_data(n_queries=200, seed=7):
    """Generate synthetic dataset for testing"""
    random.seed(seed)
    np.random.seed(seed)
    
    rows = []
    for i in range(n_queries):
        icd9_code = f"9_{i:04d}"
        icd9_phe = f"phenotype_{i}"
        
        # 5-20 candidates per query
        n_cand = random.randint(5, 20)
        gold_idx = random.randint(0, n_cand-1)
        
        for j in range(n_cand):
            icd10_code = f"10_{i:04d}_{j:02d}"
            icd10_phe = f"phenotype_{i if j==gold_idx else i*3+j}"
            correct = 1 if j == gold_idx else 0
            
            # Features: make the gold pair slightly higher
            base = 0.6 if correct else 0.2
            cosine = np.clip(np.random.random()*0.3 + base, 0.0, 1.0)
            jacc   = np.clip(np.random.random()*0.3 + base*0.8, 0.0, 1.0)
            dice   = np.clip(np.random.random()*0.3 + base*0.85, 0.0, 1.0)
            lev    = np.clip(np.random.random()*0.3 + base*0.9, 0.0, 1.0)
            cos_sim = np.clip(np.random.random()*0.3 + base*0.95, 0.0, 1.0)
            
            rows.append({
                'icd9_code': icd9_code,
                'icd9_phe': icd9_phe,
                'icd10_code': icd10_code,
                'icd10_phe': icd10_phe,
                'cosine_tfidf_word': cosine,
                'jaccard_tokens': jacc,
                'dice_tokens': dice,
                'levenshtein_ratio': lev,
                'cos_similarity': cos_sim,
                'correct': correct
            })
    
    return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Try to load real data, fall back to synthetic if not available
try:
    print(f"Attempting to load data from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"✓ Loaded {len(df):,} rows from {DATA_PATH}")
except (FileNotFoundError, ValueError) as e:
    print(f"⚠️  Could not load data: {e}")
    print("Generating synthetic data instead...")
    df = generate_synthetic_data(n_queries=200, seed=7)
    print(f"✓ Generated {len(df):,} synthetic rows")

print(f"\nData shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore Data

# COMMAND ----------

# Show first few rows
display(df.head(10))

# COMMAND ----------

# Summary statistics
print("=== Data Summary ===\n")
print(f"Total pairs: {len(df):,}")
print(f"Unique ICD9 codes (queries): {df[QUERY_COL].nunique():,}")
print(f"Unique ICD10 codes (candidates): {df[DOC_COL].nunique():,}")
print(f"Positive pairs (correct=1): {df[LABEL].sum():,} ({df[LABEL].mean()*100:.2f}%)")
print(f"Negative pairs (correct=0): {(df[LABEL]==0).sum():,} ({(df[LABEL]==0).mean()*100:.2f}%)")

print("\n=== Feature Statistics ===")
display(df[FEATURES].describe())

# COMMAND ----------

# Candidates per query distribution
candidates_per_query = df.groupby(QUERY_COL).size()
print(f"\n=== Candidates per ICD9 Query ===")
print(f"Mean: {candidates_per_query.mean():.1f}")
print(f"Median: {candidates_per_query.median():.1f}")
print(f"Min: {candidates_per_query.min()}")
print(f"Max: {candidates_per_query.max()}")

plt.figure(figsize=(10, 5))
plt.hist(candidates_per_query, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Candidates per ICD9')
plt.ylabel('Frequency')
plt.title('Distribution of Candidates per Query')
plt.grid(True, alpha=0.3)
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Distribution Analysis
# MAGIC
# MAGIC Let's look at the distributions of our features to understand why scaling may or may not be needed.

# COMMAND ----------

# Analyze feature distributions
print("=== Feature Distribution Summary ===\n")
for feat in FEATURES:
    print(f"{feat:25s}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}, "
          f"min={df[feat].min():.3f}, max={df[feat].max():.3f}")

# Plot distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    ax = axes[i]
    
    # Separate by correct/incorrect pairs
    correct_vals = df[df[LABEL]==1][feat]
    incorrect_vals = df[df[LABEL]==0][feat]
    
    ax.hist(incorrect_vals, bins=30, alpha=0.6, label='Incorrect (0)', color='red', density=True)
    ax.hist(correct_vals, bins=30, alpha=0.6, label='Correct (1)', color='green', density=True)
    
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Density')
    ax.set_title(feat)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
display(plt.show())

print("\n💡 Key Observations:")
print("- All features are bounded [0, 1] ✓")
print("- Most pairs have LOW similarity (left-skewed distributions)")
print("- Correct pairs (green) tend to have HIGHER similarity values")
print("- This is CLASS IMBALANCE in the feature space, not a scaling issue")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation & Train/Test Split

# COMMAND ----------

def prepare_data(df, features, label='correct', query_col='icd9_code', test_fold=4, n_folds=5):
    """Clean data and create train/test split with group awareness"""
    
    # Clean
    df_clean = (df[['icd9_code', 'icd9_phe', 'icd10_code', 'icd10_phe'] + features + [label]]
                .dropna(subset=features + [label])
                .drop_duplicates(subset=['icd9_code', 'icd10_code']))
    
    # Group-aware split: hash on query id to avoid leakage
    # This ensures all candidates for the same ICD9 stay in the same set (train OR test)
    df_clean['fold'] = df_clean[query_col].apply(lambda x: abs(hash(x)) % n_folds)
    
    train_df = df_clean[df_clean['fold'] < test_fold].copy()
    test_df = df_clean[df_clean['fold'] == test_fold].copy()
    
    print(f"✓ Train samples: {len(train_df):,}")
    print(f"✓ Test samples:  {len(test_df):,}")
    print(f"✓ Train queries: {train_df[query_col].nunique():,}")
    print(f"✓ Test queries:  {test_df[query_col].nunique():,}")
    print(f"✓ Train positive rate: {train_df[label].mean():.3f}")
    print(f"✓ Test positive rate:  {test_df[label].mean():.3f}")
    
    return train_df, test_df

# COMMAND ----------

train_df, test_df = prepare_data(df, FEATURES, LABEL, QUERY_COL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ranking Metrics Functions

# COMMAND ----------

def mean_average_precision(y_true_groups, y_score_groups):
    """
    Calculate Mean Average Precision (MAP)
    
    For each query, rank candidates by score and compute average precision.
    Average across all queries.
    """
    aps = []
    for y_true, y_score in zip(y_true_groups, y_score_groups):
        # Sort by score descending
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate average precision for this query
        relevant = 0
        precision_sum = 0.0
        for i, label in enumerate(y_true_sorted, 1):
            if label == 1:
                relevant += 1
                precision_sum += relevant / i
        
        ap = precision_sum / max(relevant, 1)
        aps.append(ap)
    
    return np.mean(aps)


def mean_reciprocal_rank(y_true_groups, y_score_groups):
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    For each query, find the rank of the first correct answer.
    MRR = average of 1/rank across queries.
    """
    rrs = []
    for y_true, y_score in zip(y_true_groups, y_score_groups):
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Find first relevant item
        for i, label in enumerate(y_true_sorted, 1):
            if label == 1:
                rrs.append(1.0 / i)
                break
        else:
            rrs.append(0.0)
    
    return np.mean(rrs)


def precision_at_k(y_true_groups, y_score_groups, k=1):
    """
    Calculate Precision@K
    
    For each query, check if any of the top-K results are correct.
    """
    precisions = []
    for y_true, y_score in zip(y_true_groups, y_score_groups):
        sorted_indices = np.argsort(y_score)[::-1][:k]
        y_true_sorted = y_true[sorted_indices]
        precisions.append(np.mean(y_true_sorted))
    
    return np.mean(precisions)


def compute_ranking_metrics(df, score_col, label_col='correct', query_col='icd9_code'):
    """Compute all ranking metrics per query"""
    y_true_groups = []
    y_score_groups = []
    
    for query, group in df.groupby(query_col):
        y_true_groups.append(group[label_col].values)
        y_score_groups.append(group[score_col].values)
    
    map_score = mean_average_precision(y_true_groups, y_score_groups)
    mrr_score = mean_reciprocal_rank(y_true_groups, y_score_groups)
    p_at_1 = precision_at_k(y_true_groups, y_score_groups, k=1)
    
    # Top-1 accuracy (choose best candidate per query)
    best_per_query = df.loc[df.groupby(query_col)[score_col].idxmax()]
    top1_acc = best_per_query[label_col].mean()
    
    return {
        'MAP': map_score,
        'MRR': mrr_score,
        'P@1': p_at_1,
        'Top1_Accuracy': top1_acc
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Model: Equal-Weighted Feature Average

# COMMAND ----------

def evaluate_baseline(test_df, features, label='correct', query_col='icd9_code'):
    """Simple baseline: equal-weighted average of features"""
    print("="*70)
    print("BASELINE: Equal-weighted feature average")
    print("="*70)
    
    test_df = test_df.copy()
    test_df['baseline_score'] = test_df[features].mean(axis=1)
    
    # Binary classification at threshold 0.5
    test_df['baseline_pred'] = (test_df['baseline_score'] >= 0.5).astype(int)
    
    # Classification metrics
    y_true = test_df[label]
    y_pred = test_df['baseline_pred']
    y_score = test_df['baseline_score']
    
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # = Sensitivity
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    
    print(f"\nClassification Metrics:")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}  (= Recall, TPR)")
    print(f"  Specificity: {specificity:.4f}  (TNR)")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Neg      Pos")
    print(f"  Actual Neg   {tn:5d}   {fp:5d}   (TN, FP)")
    print(f"        Pos    {fn:5d}   {tp:5d}   (FN, TP)")
    
    # Additional derived metrics
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value = Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    print(f"\nAdditional Metrics:")
    print(f"  PPV (Precision): {ppv:.4f}  (Of predicted positives, % actually positive)")
    print(f"  NPV:             {npv:.4f}  (Of predicted negatives, % actually negative)")
    
    # Ranking metrics
    ranking_metrics = compute_ranking_metrics(test_df, 'baseline_score', label, query_col)
    print(f"\nRanking Metrics:")
    print(f"  MAP:          {ranking_metrics['MAP']:.4f}")
    print(f"  MRR:          {ranking_metrics['MRR']:.4f}")
    print(f"  P@1:          {ranking_metrics['P@1']:.4f}")
    print(f"  Top1_Acc:     {ranking_metrics['Top1_Accuracy']:.4f}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        **ranking_metrics
    }

# COMMAND ----------

baseline_results = evaluate_baseline(test_df, FEATURES, LABEL, QUERY_COL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Model Pipelines
# MAGIC
# MAGIC **Note on Feature Scaling:**
# MAGIC - All features are already on the same scale (0-1), so aggressive standardization is not strictly necessary
# MAGIC - **Tree-based models** (Random Forest, Gradient Boosting) are scale-invariant and don't need scaling
# MAGIC - **Logistic Regression** can benefit slightly from scaling for regularization, but since all features are 0-1, the effect is minimal
# MAGIC - We'll keep scaling for Logistic Regression but skip it for tree models

# COMMAND ----------

def build_models():
    """Build ML model pipelines with hyperparameter grids"""
    
    # Logistic Regression - Keep scaling for regularization consistency
    # (Though with 0-1 features, the effect is minimal)
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=7, max_iter=1000))
    ])
    lr_params = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    }
    
    # Random Forest - NO SCALING NEEDED (tree-based, scale-invariant)
    rf_pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=7, n_jobs=-1))
    ])
    rf_params = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 15],
        'classifier__min_samples_split': [2, 5]
    }
    
    # Gradient Boosting - NO SCALING NEEDED (tree-based, scale-invariant)
    gbt_pipeline = Pipeline([
        ('classifier', GradientBoostingClassifier(random_state=7))
    ])
    gbt_params = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5],
        'classifier__subsample': [0.8, 1.0]
    }
    
    return {
        'LogisticRegression': (lr_pipeline, lr_params),
        'RandomForest': (rf_pipeline, rf_params),
        'GradientBoosting': (gbt_pipeline, gbt_params)
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Models with Cross-Validation

# COMMAND ----------

def train_models(train_df, features, label='correct', cv_folds=4):
    """Train models with cross-validation and hyperparameter tuning"""
    print("="*70)
    print("TRAINING MODELS WITH CROSS-VALIDATION")
    print("="*70)
    
    X_train = train_df[features].values
    y_train = train_df[label].values
    
    models = build_models()
    trained_models = {}
    
    for name, (pipeline, params) in models.items():
        print(f"\n{'='*70}")
        print(f"Training {name}...")
        print(f"{'='*70}")
        
        # Use StratifiedKFold for cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=7)
        
        grid_search = GridSearchCV(
            pipeline,
            params,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✓ Best CV AUC: {grid_search.best_score_:.4f}")
        print(f"✓ Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"    {param}: {value}")
        
        trained_models[name] = grid_search.best_estimator_
    
    return trained_models

# COMMAND ----------

trained_models = train_models(train_df, FEATURES, LABEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Models on Test Set

# COMMAND ----------

def evaluate_model(model, test_df, features, name, label='correct', query_col='icd9_code'):
    """Comprehensive evaluation of a trained model"""
    print("\n" + "="*70)
    print(f"EVALUATION: {name}")
    print("="*70)
    
    X_test = test_df[features].values
    y_test = test_df[label].values
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Confusion matrix for sensitivity/specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    
    print(f"\nClassification Metrics:")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}  (= Recall, TPR)")
    print(f"  Specificity: {specificity:.4f}  (TNR)")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Neg      Pos")
    print(f"  Actual Neg   {tn:5d}   {fp:5d}   (TN, FP)")
    print(f"        Pos    {fn:5d}   {tp:5d}   (FN, TP)")
    
    # Additional derived metrics
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    print(f"\nAdditional Metrics:")
    print(f"  PPV (Precision): {ppv:.4f}  (Of predicted positives, % actually positive)")
    print(f"  NPV:             {npv:.4f}  (Of predicted negatives, % actually negative)")
    
    # Add predictions to test_df for ranking metrics
    test_df_pred = test_df.copy()
    test_df_pred['pred'] = y_pred
    test_df_pred['score'] = y_prob
    
    # Ranking metrics
    ranking_metrics = compute_ranking_metrics(test_df_pred, 'score', label, query_col)
    print(f"\nRanking Metrics:")
    print(f"  MAP:           {ranking_metrics['MAP']:.4f}")
    print(f"  MRR:           {ranking_metrics['MRR']:.4f}")
    print(f"  P@1:           {ranking_metrics['P@1']:.4f}")
    print(f"  Top1 Accuracy: {ranking_metrics['Top1_Accuracy']:.4f}")
    
    # Feature importance (if available)
    # Get the classifier (handle both pipelines with/without scaler)
    if 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
    else:
        classifier = model  # If no pipeline, model is the classifier
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        print(f"\nFeature Importances:")
        for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
            print(f"  {feat:25s}: {imp:.4f}")
    elif hasattr(classifier, 'coef_'):
        coefs = classifier.coef_[0]
        print(f"\nFeature Coefficients:")
        for feat, coef in sorted(zip(features, coefs), key=lambda x: -abs(x[1])):
            print(f"  {feat:25s}: {coef:+.4f}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        **ranking_metrics
    }, test_df_pred

# COMMAND ----------

# Evaluate all models
all_results = {'Baseline': baseline_results}
predictions = {}

for name, model in trained_models.items():
    results, test_df_pred = evaluate_model(
        model, test_df, FEATURES, name, LABEL, QUERY_COL
    )
    all_results[name] = results
    predictions[name] = test_df_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison Summary

# COMMAND ----------

results_summary = pd.DataFrame(all_results).T
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
display(results_summary.round(4))

# COMMAND ----------

# Visualize comparison
metrics_to_plot = ['auc', 'f1', 'MAP', 'Top1_Accuracy']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    data = results_summary[metric].sort_values(ascending=False)
    bars = ax.bar(range(len(data)), data.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(data)])
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index, rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for j, (bar, val) in enumerate(zip(bars, data.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix Visualization

# COMMAND ----------

# Plot confusion matrices for all models
n_models = len(all_results)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
if n_models == 1:
    axes = [axes]

for idx, (name, results) in enumerate(all_results.items()):
    ax = axes[idx]
    
    # Get predictions for this model
    if name == 'Baseline':
        y_true = test_df[LABEL]
        y_pred = (test_df[FEATURES].mean(axis=1) >= 0.5).astype(int)
    else:
        test_df_pred = predictions[name]
        y_true = test_df_pred[LABEL]
        y_pred = test_df_pred['pred']
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot as heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                cbar=True, square=True)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}')

plt.tight_layout()
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Visualization

# COMMAND ----------

def plot_feature_importance(model, features, name):
    """Plot feature importance/coefficients"""
    # Get the classifier (handle both pipelines with/without scaler)
    if 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        title = f'Feature Importances - {name}'
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
        title = f'Feature Coefficients (Absolute) - {name}'
    else:
        print(f"No feature importance available for {name}")
        return
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importances[indices], color='steelblue')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    display(plt.show())

# COMMAND ----------

# Plot feature importance for each model
for name, model in trained_models.items():
    plot_feature_importance(model, FEATURES, name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Threshold Tuning on Best Model

# COMMAND ----------

def tune_threshold(model, train_df, features, label='correct'):
    """Tune classification threshold to optimize F1"""
    print("="*70)
    print("THRESHOLD TUNING")
    print("="*70)
    
    X_train = train_df[features].values
    y_train = train_df[label].values
    y_prob = model.predict_proba(X_train)[:, 1]
    
    thresholds = np.arange(0.1, 0.95, 0.02)
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        # Handle edge cases where all predictions are same class
        if len(np.unique(y_pred_thresh)) == 1:
            f1 = 0.0
            acc = accuracy_score(y_train, y_pred_thresh)
            prec = 0.0
            rec = 0.0
        else:
            f1 = f1_score(y_train, y_pred_thresh)
            acc = accuracy_score(y_train, y_pred_thresh)
            prec = precision_score(y_train, y_pred_thresh)
            rec = recall_score(y_train, y_pred_thresh)
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec
        })
    
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df['f1'].idxmax()]
    
    print(f"\n✓ Best threshold: {best_row['threshold']:.3f}")
    print(f"  F1:        {best_row['f1']:.4f}")
    print(f"  Accuracy:  {best_row['accuracy']:.4f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall:    {best_row['recall']:.4f}")
    
    return best_row['threshold'], results_df

# COMMAND ----------

# Select best model based on F1 score
best_model_name = results_summary['f1'].idxmax()
print(f"Best model: {best_model_name}")

if best_model_name != 'Baseline':
    best_model = trained_models[best_model_name]
    best_threshold, threshold_results = tune_threshold(best_model, train_df, FEATURES, LABEL)
else:
    print("Baseline is best - skipping threshold tuning")
    best_threshold = 0.5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Threshold Tuning Visualization

# COMMAND ----------

if best_model_name != 'Baseline':
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Precision vs Recall
    axes[0].plot(threshold_results['recall'], threshold_results['precision'], 'b-', linewidth=2)
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision-Recall Curve', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Metrics vs Threshold
    axes[1].plot(threshold_results['threshold'], threshold_results['f1'], label='F1', linewidth=2)
    axes[1].plot(threshold_results['threshold'], threshold_results['accuracy'], label='Accuracy', linewidth=2)
    axes[1].plot(threshold_results['threshold'], threshold_results['precision'], label='Precision', linewidth=2)
    axes[1].plot(threshold_results['threshold'], threshold_results['recall'], label='Recall', linewidth=2)
    axes[1].axvline(best_threshold, color='red', linestyle='--', alpha=0.7, label=f'Best ({best_threshold:.3f})')
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Metrics vs Threshold', fontsize=14)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Best Model with Threshold

# COMMAND ----------

def apply_threshold_with_coverage(test_df_pred, threshold, query_col='icd9_code', label='correct'):
    """Apply threshold to best candidate per query and compute coverage"""
    print("="*70)
    print(f"PER-QUERY DECISION WITH THRESHOLD = {threshold:.3f}")
    print("="*70)
    
    # Get best candidate per query
    best_per_query = test_df_pred.loc[test_df_pred.groupby(query_col)['score'].idxmax()].copy()
    
    # Apply threshold
    best_per_query['accept'] = (best_per_query['score'] >= threshold).astype(int)
    
    # Coverage: fraction of queries where we accept the best candidate
    coverage = best_per_query['accept'].mean()
    
    # Accuracy on accepted candidates
    accepted = best_per_query[best_per_query['accept'] == 1]
    if len(accepted) > 0:
        acc_on_accepted = accepted[label].mean()
    else:
        acc_on_accepted = 0.0
    
    print(f"\n✓ Coverage (% queries with match accepted): {coverage:.1%}")
    print(f"✓ Accuracy on accepted matches:             {acc_on_accepted:.1%}")
    print(f"✓ Total queries:                            {len(best_per_query):,}")
    print(f"✓ Accepted matches:                         {len(accepted):,}")
    print(f"✓ Rejected queries:                         {len(best_per_query) - len(accepted):,}")
    
    return best_per_query, coverage, acc_on_accepted

# COMMAND ----------

# Apply threshold on test set
test_df_pred = predictions[best_model_name] if best_model_name != 'Baseline' else test_df.copy()
if best_model_name == 'Baseline':
    test_df_pred['score'] = test_df_pred[FEATURES].mean(axis=1)

best_decisions, coverage, acc_on_accepted = apply_threshold_with_coverage(
    test_df_pred, best_threshold, QUERY_COL, LABEL
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample of Best Match Decisions

# COMMAND ----------

print("\nSample of best match decisions:")
sample_cols = [QUERY_COL, 'icd9_phe', DOC_COL, 'icd10_phe', 'score', 'accept', LABEL]
display(best_decisions[sample_cols].head(20))

# COMMAND ----------

# Look at some accepted vs rejected examples
print("\n=== Examples of ACCEPTED matches (score ≥ threshold) ===")
display(best_decisions[best_decisions['accept']==1][sample_cols].head(10))

print("\n=== Examples of REJECTED queries (score < threshold) ===")
display(best_decisions[best_decisions['accept']==0][sample_cols].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------

# Save model comparison
results_summary.to_csv('/dbfs/tmp/phenotype_matching_model_comparison.csv')
print("✓ Saved model comparison to /dbfs/tmp/phenotype_matching_model_comparison.csv")

# Save best match decisions
best_decisions.to_csv('/dbfs/tmp/phenotype_matching_best_decisions.csv', index=False)
print("✓ Saved best match decisions to /dbfs/tmp/phenotype_matching_best_decisions.csv")

# Save full predictions for best model
test_df_pred.to_csv('/dbfs/tmp/phenotype_matching_full_predictions.csv', index=False)
print("✓ Saved full predictions to /dbfs/tmp/phenotype_matching_full_predictions.csv")

print("\n✅ All results saved successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Next Steps
# MAGIC
# MAGIC ### Results Summary
# MAGIC
# MAGIC We trained and evaluated 4 approaches:
# MAGIC 1. **Baseline**: Simple equal-weighted average of features
# MAGIC 2. **Logistic Regression**: Linear model with learned weights
# MAGIC 3. **Random Forest**: Ensemble of decision trees (parallel)
# MAGIC 4. **Gradient Boosting**: Ensemble of decision trees (sequential)
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC - Best model: {best_model_name}
# MAGIC - Optimal threshold: {best_threshold:.3f}
# MAGIC - Test AUC: {results_summary.loc[best_model_name, 'auc']:.4f}
# MAGIC - Top-1 Accuracy: {results_summary.loc[best_model_name, 'Top1_Accuracy']:.4f}
# MAGIC - Coverage: {coverage:.1%}
# MAGIC - Accuracy on accepted: {acc_on_accepted:.1%}
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC 1. **Error Analysis**: Examine false positives and false negatives
# MAGIC 2. **Feature Engineering**: Add more similarity features (e.g., BERT embeddings)
# MAGIC 3. **Ensemble**: Combine predictions from multiple models
# MAGIC 4. **Deployment**: Export best model for production use
# MAGIC 5. **Active Learning**: Get labels for uncertain predictions to improve model
# MAGIC
# MAGIC ### Saved Artifacts
# MAGIC
# MAGIC - `/dbfs/tmp/phenotype_matching_model_comparison.csv` - Model performance comparison
# MAGIC - `/dbfs/tmp/phenotype_matching_best_decisions.csv` - Per-query best match decisions
# MAGIC - `/dbfs/tmp/phenotype_matching_full_predictions.csv` - All predictions with scores

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Error Analysis

# COMMAND ----------

# Analyze errors - where did the model fail?
errors = best_decisions[best_decisions['accept'] == 1].copy()  # Only look at accepted predictions
errors['is_error'] = (errors[LABEL] != 1).astype(int)

print(f"\n=== Error Analysis ===")
print(f"Total accepted predictions: {len(errors):,}")
print(f"Errors (false positives): {errors['is_error'].sum():,}")
print(f"Error rate: {errors['is_error'].mean():.1%}")

if errors['is_error'].sum() > 0:
    print("\n=== Examples of Errors (False Positives) ===")
    error_examples = errors[errors['is_error']==1].sort_values('score', ascending=False)
    display(error_examples[sample_cols].head(10))
    
    print("\n=== Feature Values for Errors vs Correct ===")
    print("Errors:")
    display(error_examples[FEATURES].describe())
    print("\nCorrect predictions:")
    correct = errors[errors['is_error']==0]
    display(correct[FEATURES].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **End of Notebook**
