# Phenotype Matching Analysis

This project evaluates models for matching ICD9 to ICD10 phenotype codes using similarity features.

## Files

- **`02_phenotype_matching_models_evaluation_pyspark.py`**: Original PySpark/Databricks notebook implementation
- **`02_phenotype_matching_models_evaluation_pandas.py`**: Refactored pandas/scikit-learn implementation (recommended for small datasets)
- **`requirements.txt`**: Python package dependencies

## LLM-Assisted ICD9→ICD10 Mapping (New)

This repository now includes a testable framework under `llm_mapping/` to select the best ICD10 match using an LLM with a vector store of all ICD10 codes. It enforces non-hallucination by restricting choices to an allowed set per query and validating against a pre-built ICD10 index.

### Quickstart

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Prepare an ICD10 CSV with columns:
   - `code`, `name`
   - Optional: `synonyms` (pipe-delimited), `description`

3) Build the vector store:

```bash
python -m llm_mapping.vector_store --icd10-csv /path/to/icd10.csv --out-dir /tmp/icd10_index
```

4) Prepare an input CSV with columns:
   - `icd9_code`, `icd9_name`
   - Optional: `direct_candidates` as pipe-delimited codes or `code:name` pairs

5) Run the batch mapper (requires `OPENAI_API_KEY` in env):

```bash
export OPENAI_API_KEY=... 
python -m llm_mapping.batch_runner --input /path/to/input.csv --icd10-store /tmp/icd10_index --out-jsonl results.jsonl --out-csv results.csv --model gpt-4o-mini
```

Output fields include JSON attributes: `selected_code`, `selected_name`, `confidence`, `rationale`, `mapping_category` (NONE/CLOSE_MATCH/OTHER_MATCH), `match_specificity` (EXACT/CLOSE/MORE_BROAD), `external_choice_reason` (MULTIMAP/BAD_MAPPING/N/A).

### Prompting Rules Implemented

The prompt enforces:
- Prefer direct candidates when suitable; otherwise select from retrieved universe.
- Choose broader concepts if ICD9 spans multiple specific ICD10s; avoid narrow subsets.
- Do not hallucinate codes; must pick from provided list or return no_confident_match.

You can tweak the messages in `llm_mapping/prompt_builder.py`.


## Refactored Pandas Version

The pandas version offers several advantages for small datasets:

### Key Changes

1. **Data Processing**: PySpark → pandas DataFrames
2. **ML Models**: Spark MLlib → scikit-learn
3. **Scaling**: Spark StandardScaler → sklearn StandardScaler
4. **Cross-validation**: Spark CrossValidator → sklearn GridSearchCV
5. **Metrics**: Custom implementations using numpy/sklearn
6. **Visualization**: Added matplotlib/seaborn plots

### Features

- ✅ Logistic Regression, Random Forest, Gradient Boosting classifiers
- ✅ Classification metrics: AUC, Accuracy, F1, Precision, Recall
- ✅ Ranking metrics: MAP, MRR, Precision@1, Top-1 Accuracy
- ✅ Threshold tuning with visualization
- ✅ Per-query best match selection with coverage analysis
- ✅ Feature importance visualization
- ✅ Group-aware train/test split (no query leakage)
- ✅ Baseline model for comparison
- ✅ Synthetic data generation for testing

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python 02_phenotype_matching_models_evaluation_pandas.py
```

The script will:
1. Try to load data from the configured path
2. If not found, generate synthetic data for demonstration
3. Train and evaluate multiple models
4. Output comparison results and visualizations
5. Save results to CSV files

### Outputs

- `model_comparison_results.csv`: Summary of all models
- `best_match_decisions.csv`: Per-query best matches with accept/reject decisions
- `threshold_tuning.png`: Threshold optimization curves
- `feature_importance_*.png`: Feature importance plots for each model

### Configuration

Edit the `main()` function to customize:
- Feature columns
- Data path
- Model parameters
- Train/test split ratio

## Data Format

Expected columns:
- `icd9_code`, `icd9_phe` (ICD9 code and phenotype name)
- `icd10_code`, `icd10_phe` (ICD10 code and phenotype name)
- `cosine_tfidf_word`, `jaccard_tokens`, `dice_tokens`, `levenshtein_ratio`, `cos_similarity` (similarity features)
- `correct` (binary label: 1 = correct match, 0 = incorrect)

## Performance

For small datasets (<1M rows), the pandas version is:
- **Faster** to run (no Spark overhead)
- **Easier** to debug and modify
- **More portable** (runs anywhere Python runs)
- **Better integrated** with standard ML/stats ecosystem

For large datasets (>1M rows), consider the PySpark version for distributed processing.


