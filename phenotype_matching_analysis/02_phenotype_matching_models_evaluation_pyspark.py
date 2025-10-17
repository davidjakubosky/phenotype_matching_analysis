# Databricks notebook source
# MAGIC %md
# MAGIC # Phenotype Matching: ICD9 ↔ ICD10 with Similarity Features
# MAGIC
# MAGIC This notebook trains and evaluates models that predict whether an ICD9 phenotype name matches an ICD10 phenotype name using similarity features.
# MAGIC
# MAGIC **Schema expected in `sdf`:**
# MAGIC - `icd9_code` (string), `icd9_phe` (string)
# MAGIC - `icd10_code` (string), `icd10_phe` (string)
# MAGIC - Features: `cosine_tfidf_word`, `jaccard_tokens`, `dice_tokens`, `levenshtein_ratio` (double)
# MAGIC - Label: `correct` (int: 0/1)
# MAGIC
# MAGIC Core outputs:
# MAGIC - Baselines (simple weighted score), Logistic Regression, Random Forest, GBT
# MAGIC - Classification metrics (AUC, Accuracy, F1)
# MAGIC - Ranking metrics per *query* (MAP, MRR, Precision@1 / Top-1 accuracy)
# MAGIC - Threshold tuning and per-query decisioning (choose best candidate per ICD9)
# MAGIC
# MAGIC **Note**: If you don't already have `sdf` in the workspace, run the synthetic data cell below to generate a small demo dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import RankingMetrics
import mlflow
import mlflow.spark
from pyspark.sql import Window
# Reproducibility
spark.conf.set("spark.sql.shuffle.partitions", spark.conf.get("spark.sql.shuffle.partitions", "200"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) Build a tiny synthetic dataset if `sdf` is absent

# COMMAND ----------

if 'sdf' not in locals():
    import random
    random.seed(7)
    n_queries = 200
    rows = []
    for i in range(n_queries):
        icd9_code = f"9_{i:04d}"
        icd9_phe = f"phenotype_{i}"
        # 5-20 candidates per query
        n_cand = random.randint(5, 20)
        # pick a gold index
        gold_idx = random.randint(0, n_cand-1)
        for j in range(n_cand):
            icd10_code = f"10_{i:04d}_{j:02d}"
            icd10_phe = f"phenotype_{i if j==gold_idx else i*3+j}"
            correct = 1 if j == gold_idx else 0
            # Features: make the gold pair slightly higher
            base = 0.6 if correct else 0.2
            cosine = min(1.0, max(0.0, random.random()*0.3 + base))
            jacc   = min(1.0, max(0.0, random.random()*0.3 + base*0.8))
            dice   = min(1.0, max(0.0, random.random()*0.3 + base*0.85))
            lev    = min(1.0, max(0.0, random.random()*0.3 + base*0.9))
            rows.append((icd9_code, icd9_phe, icd10_code, icd10_phe, float(cosine), float(jacc), float(dice), float(lev), None, None, int(correct)))
    schema = T.StructType([
        T.StructField('icd9_code', T.StringType()),
        T.StructField('icd9_phe', T.StringType()),
        T.StructField('icd10_code', T.StringType()),
        T.StructField('icd10_phe', T.StringType()),
        T.StructField('cosine_tfidf_word', T.DoubleType()),
        T.StructField('jaccard_tokens', T.DoubleType()),
        T.StructField('dice_tokens', T.DoubleType()),
        T.StructField('levenshtein_ratio', T.DoubleType()),
        T.StructField('correct_mapped_code', T.StringType()),
        T.StructField('correct_mapped_phe', T.StringType()),
        T.StructField('correct', T.IntegerType()),
    ])
    sdf = spark.createDataFrame(rows, schema)

# Sanity check
display(sdf.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess & Group-Aware Split (avoid leakage across the same ICD9 query)

# COMMAND ----------

fn = '/mnt/data/projects/similarity/AoU_test_set_pairwise_w_correct_answers.delta'
sdf = spark.read.load(fn)

# COMMAND ----------

FEATURES = [
    'cosine_tfidf_word',
    'jaccard_tokens',
    'dice_tokens',
    'levenshtein_ratio',
    'cos_similarity'
]
LABEL = 'correct'
QUERY_COL = 'icd9_code'   # each ICD9 has many candidates (ICD10)
DOC_COL   = 'icd10_code'

# Clean
sdf_clean = (sdf
    .select('icd9_code','icd9_phe','icd10_code','icd10_phe', *FEATURES, LABEL)
    .dropna(subset=FEATURES+[LABEL])
    .dropDuplicates(['icd9_code','icd10_code'])
)

# Group-aware fold: hash on query id
sdf_clean = sdf_clean.withColumn('fold', (F.abs(F.hash(F.col(QUERY_COL))) % F.lit(5)).cast('int'))
train_df = sdf_clean.filter(F.col('fold') < 4)
test_df  = sdf_clean.filter(F.col('fold') == 4)

print('Train rows:', train_df.count(), ' Test rows:', test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline: Simple Weighted Score & per-query Top-1 selection

# COMMAND ----------

# Start with equal weights; later we can tune via logistic regression
w = {f: 1.0 for f in FEATURES}

expr = sum([F.col(f)*w[f] for f in FEATURES])
with_scores = test_df.withColumn('baseline_score', expr)

# Per-query choose best candidate by score
w_best = (with_scores
    .withColumn('rn', F.row_number().over(
        Window.partitionBy(QUERY_COL).orderBy(F.col('baseline_score').desc())
    ))
    .filter(F.col('rn') == 1)
)

from pyspark.sql import Window

# Top-1 accuracy (Precision@1)
acc_top1 = w_best.agg(F.avg(F.col(LABEL).cast('double')).alias('p_at_1')).collect()[0]['p_at_1']
print('Baseline P@1 (Top-1 Accuracy):', acc_top1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Pipeline: Assemble → Scale → Classifier

# COMMAND ----------

assembler = VectorAssembler(inputCols=FEATURES, outputCol='features_raw')
scaler = StandardScaler(inputCol='features_raw', outputCol='features', withMean=True, withStd=True)

lr = LogisticRegression(featuresCol='features', labelCol=LABEL, probabilityCol='prob', rawPredictionCol='rawPred', predictionCol='pred')
rf = RandomForestClassifier(featuresCol='features', labelCol=LABEL, probabilityCol='prob', rawPredictionCol='rawPred', predictionCol='pred', seed=7)
gbt = GBTClassifier(featuresCol='features', labelCol=LABEL, predictionCol='pred', maxIter=60, stepSize=0.05, seed=7)

pipelines = {
    'lr': Pipeline(stages=[assembler, scaler, lr]),
    'rf': Pipeline(stages=[assembler, scaler, rf]),
    'gbt': Pipeline(stages=[assembler, scaler, gbt])
}

param_grids = {
    'lr': (ParamGridBuilder()
           .addGrid(lr.regParam, [0.0, 0.01, 0.1])
           .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
           .build()),
    'rf': (ParamGridBuilder()
           .addGrid(rf.numTrees, [50, 200])
           .addGrid(rf.maxDepth, [5, 10])
           .build()),
    'gbt': (ParamGridBuilder()
            .addGrid(gbt.maxDepth, [3, 5])
            .build())
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross-Validation on the Training Set

# COMMAND ----------

auc_eval = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol='rawPred', metricName='areaUnderROC')

cv_models = {}
for name, pipe in pipelines.items():
    print(f"Training {name}…")
    with mlflow.start_run(run_name=f"phenotype_match_{name}"):
        cv = CrossValidator(estimator=pipe,
                            estimatorParamMaps=param_grids[name],
                            evaluator=auc_eval,
                            numFolds=4,
                            parallelism=4,
                            seed=7)
        cv_model = cv.fit(train_df)
        cv_models[name] = cv_model
        best_auc = cv_model.avgMetrics[cv_model.avgMetrics.index(max(cv_model.avgMetrics))]
        print(f"Best CV AUC ({name}): {best_auc:.4f}")
        mlflow.log_metric("cv_best_auc", float(best_auc))
        mlflow.spark.log_model(cv_model.bestModel, artifact_path=f"{name}_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation: Classification & Ranking (per-query)

# COMMAND ----------

from pyspark.sql import Window

def evaluate_model(cv_model, name):
    print(f"\n=== Evaluation for {name} ===")
    pred = cv_model.transform(test_df)

    # Classification metrics
    auc = auc_eval.evaluate(pred)
    acc = MulticlassClassificationEvaluator(labelCol=LABEL, predictionCol='pred', metricName='accuracy').evaluate(pred)
    f1  = MulticlassClassificationEvaluator(labelCol=LABEL, predictionCol='pred', metricName='f1').evaluate(pred)
    print(f"AUC={auc:.4f}  ACC={acc:.4f}  F1={f1:.4f}")

    # Ranking metrics per query: use predicted probability/score to rank candidates per ICD9
    # 1) Collect predictions per-query into lists of doc ids sorted by score
    score_col = 'prob' if 'prob' in pred.columns else 'rawPred'
    pred_scored = pred.withColumn('score', F.col(score_col).getItem(1) if score_col=='prob' else F.col(score_col)[1])

    w = Window.partitionBy(QUERY_COL).orderBy(F.col('score').desc())
    ranked = pred_scored.withColumn('rank', F.row_number().over(w))

    # For RankingMetrics: (predictedRanking, groundTruth)
    # predictedRanking: list of doc ids in predicted order
    # groundTruth: list with the single correct doc id (or multiple if present)
    pred_lists = (ranked
        .groupBy(QUERY_COL)
        .agg(F.collect_list(F.struct('rank', F.col(DOC_COL))).alias('candidates'),
             F.collect_set(F.when(F.col(LABEL)==1, F.col(DOC_COL))).alias('truth_set'))
        .select(QUERY_COL,
                F.expr("transform(array_sort(candidates), x -> x.icd10_code)").alias('predicted_ranking'),
                'truth_set')
    )

    # Convert to RDD[(List[str], List[str])]
    rdd = pred_lists.select('predicted_ranking','truth_set').rdd.map(lambda r: (r[0], list(r[1])))
    rm = RankingMetrics(rdd)

    mapk = rm.meanAveragePrecision
    prec1 = rm.precisionAt(1)
    mrr  = rm.meanReciprocalRank
    print(f"Ranking: MAP={mapk:.4f}  P@1={prec1:.4f}  MRR={mrr:.4f}")

    # Per-query choose best candidate and compute Top-1 accuracy
    top1 = (ranked.filter(F.col('rank')==1)
            .agg(F.avg(F.col(LABEL).cast('double')).alias('p_at_1'))
            .collect()[0]['p_at_1'])
    print(f"Top-1 (choose best candidate): {top1:.4f}")

    return {
        'auc': auc,
        'acc': acc,
        'f1': f1,
        'map': mapk,
        'p_at_1': prec1,
        'mrr': mrr,
        'top1_choose_best': top1
    }

results = {}
for name, model in cv_models.items():
    results[name] = evaluate_model(model, name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Threshold Tuning & Calibrated Decisioning
# MAGIC
# MAGIC For a deployment that classifies pairs as match/non-match, tune the threshold on the predicted probability to optimize F1 or desired precision/recall.

# COMMAND ----------

from pyspark.sql import Row

def tune_threshold(cv_model, metric='f1'):
    pred = cv_model.transform(train_df)
    score_col = 'prob' if 'prob' in pred.columns else 'rawPred'
    pred = pred.withColumn('score', F.col(score_col).getItem(1) if score_col=='prob' else F.col(score_col)[1])

    # Evaluate several thresholds
    thresholds = [i/100 for i in range(10, 91, 2)]
    rows = []
    for t in thresholds:
        labeled = pred.withColumn('pred_label', (F.col('score') >= F.lit(t)).cast('int'))
        acc = MulticlassClassificationEvaluator(labelCol=LABEL, predictionCol='pred_label', metricName='accuracy').evaluate(labeled)
        f1  = MulticlassClassificationEvaluator(labelCol=LABEL, predictionCol='pred_label', metricName='f1').evaluate(labeled)
        rows.append(Row(threshold=t, accuracy=acc, f1=f1))
    return spark.createDataFrame(rows)

# Example: tune on LR
if 'lr' in cv_models:
    thresh_df = tune_threshold(cv_models['lr'])
    display(thresh_df.orderBy(F.col('f1').desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Query Best Match Strategy with Threshold Safeguard
# MAGIC
# MAGIC We often need exactly-one best match per ICD9. We rank candidates by score, take the best, and optionally require the score to exceed a threshold to accept; otherwise, return `None`.

# COMMAND ----------

BEST_MODEL = cv_models[list(cv_models.keys())[0]]  # pick first; adjust to your choice

pred = BEST_MODEL.transform(test_df)
score_col = 'prob' if 'prob' in pred.columns else 'rawPred'
pred = pred.withColumn('score', F.col(score_col).getItem(1) if score_col=='prob' else F.col(score_col)[1])

from pyspark.sql import Window
w = Window.partitionBy(QUERY_COL).orderBy(F.col('score').desc())
ranked = pred.withColumn('rank', F.row_number().over(w))

THRESH = 0.5  # adjust based on the threshold tuning above
best_decision = (ranked
    .filter(F.col('rank')==1)
    .withColumn('accept', (F.col('score') >= F.lit(THRESH)).cast('int'))
    .select(QUERY_COL, 'icd9_phe', DOC_COL, 'icd10_phe', 'score', 'accept', LABEL)
)

# If `accept`==0, treat as no-match for that query. Compute conditional accuracy on accepted.
accepted = best_decision.filter(F.col('accept')==1)
coverage = accepted.count() / best_decision.count()
acc_on_accepted = accepted.agg(F.avg(F.col(LABEL).cast('double'))).collect()[0][0]
print(f"Coverage={coverage:.3f}  Accuracy(on accepted)={acc_on_accepted:.3f}")

display(best_decision.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Optional) Global One-to-One Consistency via Assignment
# MAGIC
# MAGIC If you require a *global* one-to-one mapping (each ICD9 and ICD10 used at most once), solve a maximum-weight bipartite matching (Hungarian algorithm) on a block (e.g., same chapter or first 3 chars). In Spark, compute scores, then per-block collect to a Pandas UDF and run `scipy.optimize.linear_sum_assignment`.
# MAGIC
# MAGIC **Sketch:**
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC from scipy.optimize import linear_sum_assignment
# MAGIC
# MAGIC @pandas_udf("icd9_code string, icd10_code string, score double", PandasUDFType.GROUPED_MAP)
# MAGIC def assign_block(pdf: pd.DataFrame) -> pd.DataFrame:
# MAGIC     rows = sorted(pdf['icd9_code'].unique())
# MAGIC     cols = sorted(pdf['icd10_code'].unique())
# MAGIC     M = -1e3*np.ones((len(rows), len(cols)))  # large negative for missing
# MAGIC     for _, r in pdf.iterrows():
# MAGIC         i = rows.index(r.icd9_code); j = cols.index(r.icd10_code)
# MAGIC         M[i,j] = -r.score  # minimize cost = negative score
# MAGIC     rr, cc = linear_sum_assignment(M)
# MAGIC     out = [(rows[i], cols[j], -M[i,j]) for i,j in zip(rr,cc) if M[i,j] < 0]
# MAGIC     return pd.DataFrame(out, columns=['icd9_code','icd10_code','score'])
# MAGIC
# MAGIC # usage: ranked.groupby('block').apply(assign_block)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## What else to try (ideas)
# MAGIC
# MAGIC 1. **Better Features / Embeddings**: Add SBERT/BioBERT sentence embeddings of phenotype names and use cosine similarity as another feature (two-tower retrieval). 
# MAGIC 2. **Learning-to-Rank**: Use pairwise/listwise rankers (e.g., LambdaMART via LightGBM/SynapseML) trained to optimize MAP/MRR directly.
# MAGIC 3. **Two-Stage Pipeline**: Fast candidate generation (token Jaccard/TF-IDF) → re-rank with heavier model (GBTs or cross-encoder).
# MAGIC 4. **Blocking**: Limit candidates by shared tokens, prefix/suffix rules, or ICD chapter to reduce noise and speed.
# MAGIC 5. **Calibration**: Post-hoc isotonic calibration of tree model scores for thresholdable probabilities.
# MAGIC 6. **Class Imbalance**: If negatives dominate, use class weights, downsampling, or focal loss (outside vanilla Spark ML).
# MAGIC 7. **Active Learning**: Surface top-uncertain pairs for manual review to improve labels.
# MAGIC 8. **Data QA**: Ensure `correct` at most once per query; fix any multi-positive anomalies or score them accordingly in ranking metrics.
# MAGIC
# MAGIC ---
# MAGIC **End of notebook.**