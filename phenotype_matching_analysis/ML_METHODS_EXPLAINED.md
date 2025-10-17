# 🎓 Machine Learning Methods Explained (High School Level)

## 1️⃣ Logistic Regression - "The Weighted Voting System"

### The Intuition
Imagine you're deciding if two medical codes match based on 5 clues (features). Logistic Regression learns how important each clue is and combines them with a formula:

```
Score = w1×cosine + w2×jaccard + w3×dice + w4×levenshtein + w5×cos_sim + bias
```

Each feature gets a **weight** (w1, w2, etc.). Positive weights mean "this helps indicate a match", negative weights mean "this indicates no match".

### How It Works
1. Start with random weights
2. Look at training examples and see if predictions are right
3. Adjust weights to reduce errors (like adjusting recipe ingredients)
4. Repeat until weights are as good as possible

### The Math Trick
The score gets passed through a **sigmoid function** that squishes it between 0 and 1:
```
Probability = 1 / (1 + e^(-score))
```
This converts the score into a probability: "I'm 73% confident this is a match"

### Hyperparameter: C (regularization)
**Problem:** The model might "memorize" training data instead of learning general patterns (called overfitting).

**Solution:** The C parameter controls this:
- **Small C (0.01)**: Be cautious, keep weights small → less overfitting, simpler model
- **Large C (10.0)**: Be bold, allow large weights → more complex model, fits training data closely

We test multiple C values and pick the best!

### Pros & Cons
✅ Fast to train
✅ Easy to interpret (can see which features matter most)
✅ Works well for linearly separable problems
❌ Can't learn complex non-linear patterns
❌ Assumes features combine linearly

---

## 2️⃣ Random Forest - "The Committee of Decision Trees"

### The Intuition
Instead of one opinion, ask 100 "experts" (decision trees) and take a vote!

### What's a Decision Tree?
It's like a flowchart of yes/no questions:
```
Is cosine_similarity > 0.6?
├─ YES → Is jaccard > 0.5?
│  ├─ YES → MATCH! ✅
│  └─ NO → Is dice > 0.4?
│     ├─ YES → MATCH! ✅
│     └─ NO → NO MATCH ❌
└─ NO → NO MATCH ❌
```

The tree learns the best questions to ask and in what order.

### How Random Forest Works
1. Create many trees (50-200 of them)
2. Each tree is trained on a random subset of the data
3. Each tree split considers only a random subset of features
4. For a new example, each tree votes
5. Final prediction = majority vote

**Why random?** 
- Prevents all trees from being identical
- Like asking diverse people instead of clones
- Reduces overfitting through "wisdom of crowds"

### Hyperparameters

**n_estimators** (number of trees):
- 50 trees: Fast but less accurate
- 200 trees: Slower but more stable predictions
- More trees = better (usually), but diminishing returns after ~200

**max_depth** (how deep each tree can grow):
- Depth 5: Shallow trees, simple patterns only
- Depth 15: Deep trees, can learn complex patterns
- Too deep → overfitting (memorizing noise)
- Too shallow → underfitting (missing patterns)

**min_samples_split** (minimum data to split a node):
- 2: Split aggressively (more complex trees)
- 5: Need more evidence to split (simpler trees)
- Prevents splitting on tiny groups (reduces noise)

### Pros & Cons
✅ Can learn complex non-linear patterns
✅ Robust to outliers
✅ Handles features of different scales well
✅ Built-in feature importance
❌ Can overfit if not tuned properly
❌ Less interpretable than logistic regression
❌ Slower to train and predict

---

## 3️⃣ Gradient Boosting - "The Iterative Learner"

### The Intuition
Instead of training trees in parallel (Random Forest), train them **sequentially**, where each new tree fixes the mistakes of the previous ones.

Think of it like studying for a test:
1. Take a practice test, get 70%
2. Study the questions you got wrong
3. Take another test focusing on weak areas, get 85%
4. Repeat until you ace it!

### How It Works
1. Start with a simple prediction (like "average")
2. Train a small tree to predict the **errors** from step 1
3. Add this tree's predictions to the original
4. Train another tree to predict the **remaining errors**
5. Repeat 50-100 times

Each tree is **weak** (shallow, simple) but together they're **strong**.

### The Math
```
Prediction_v1 = Tree1
Prediction_v2 = Tree1 + (learning_rate × Tree2)
Prediction_v3 = Tree1 + (LR × Tree2) + (LR × Tree3)
...
Final = Tree1 + LR×Tree2 + LR×Tree3 + ... + LR×Tree100
```

### Hyperparameters

**n_estimators** (number of boosting rounds):
- 50: Quick baseline
- 100: More accurate
- More rounds = better fit to training data
- Too many → overfitting

**learning_rate** (how fast to learn):
- 0.05: Slow and steady (more rounds needed, but more stable)
- 0.1: Standard choice
- 0.3: Fast learning (fewer rounds, risk of overfitting)
- Lower learning rate + more estimators = better (but slower)

**max_depth** (tree depth):
- 3: Very shallow "stumps" (safe, less overfitting)
- 5: Medium complexity
- Boosting works best with shallow trees (unlike Random Forest)

**subsample** (fraction of data per tree):
- 0.8: Use 80% of data randomly for each tree (like mini-batches)
- 1.0: Use all data
- <1.0 adds randomness, reduces overfitting, speeds up training

### Pros & Cons
✅ Often the most accurate method
✅ Can learn very complex patterns
✅ Good feature importance
✅ Handles mixed feature types well
❌ Slower to train (sequential, not parallel)
❌ More prone to overfitting than Random Forest
❌ More hyperparameters to tune
❌ Less interpretable

---

## 🎯 Which Model to Choose?

**Start with Logistic Regression if:**
- You want interpretability
- You have limited data
- Speed is critical
- Features are already good

**Use Random Forest if:**
- You have non-linear patterns
- You want robustness
- You have enough data
- You want easy parallelization

**Use Gradient Boosting if:**
- You want maximum accuracy
- You have time to tune hyperparameters
- You can afford longer training
- You're entering a competition! 😄

**In practice:** Try all three and compare!

---

## 🔍 Cross-Validation: The Fair Testing Strategy

### The Problem
If you tune hyperparameters using the test set, you're "peeking at the answers"!

### The Solution: Cross-Validation
Split training data into 4 pieces (folds):

```
Fold 1: [ Test ] [ Train ] [ Train ] [ Train ]  → Measure accuracy
Fold 2: [ Train ] [ Test ] [ Train ] [ Train ]  → Measure accuracy
Fold 3: [ Train ] [ Train ] [ Test ] [ Train ]  → Measure accuracy
Fold 4: [ Train ] [ Train ] [ Train ] [ Test ]  → Measure accuracy

Average accuracy across all 4 folds = CV score
```

**Why 4 folds?**
- More folds = more reliable estimate, but slower
- 4-5 folds is a good balance

### Grid Search
We try many hyperparameter combinations:
```
Logistic Regression:
  C = 0.01 → CV score = 0.82
  C = 0.1  → CV score = 0.87
  C = 1.0  → CV score = 0.89 ← BEST
  C = 10.0 → CV score = 0.88
```

Pick the combination with the best CV score!

---

## 📊 Evaluation Metrics Explained

### Classification Metrics (Individual Predictions)

**Accuracy:**
```
Accuracy = (Correct predictions) / (Total predictions)
Example: 85 correct out of 100 = 85%
```
Simple but can be misleading if classes are imbalanced!

**Precision:**
```
Precision = True Positives / (True Positives + False Positives)
          = "Of all matches I predicted, how many were actually correct?"
```
Example: I said 100 pairs match, but only 90 truly match → 90% precision
**Useful when:** False alarms are costly

**Recall:**
```
Recall = True Positives / (True Positives + False Negatives)
       = "Of all actual matches, how many did I find?"
```
Example: There are 100 true matches, I found 85 → 85% recall
**Useful when:** Missing matches is costly

**F1 Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
The "harmonic mean" of precision and recall - balances both!

**AUC (Area Under ROC Curve):**
- Measures how well the model separates classes across all possible thresholds
- 0.5 = random guessing (flip a coin)
- 1.0 = perfect separation
- 0.8+ = good model

### Ranking Metrics (For Search Results)

Remember: Each ICD9 has multiple ICD10 candidates ranked by score.

**Precision@1 (P@1):**
```
P@1 = "How often is the top-ranked candidate correct?"
```
Example: For 100 queries, top result is correct 73 times → P@1 = 73%

**Mean Reciprocal Rank (MRR):**
```
If correct answer is rank 1 → score = 1.0
If correct answer is rank 2 → score = 0.5
If correct answer is rank 3 → score = 0.33
If correct answer is rank 4 → score = 0.25
...
MRR = average of these scores
```
Example: 3 queries with correct at ranks [1, 2, 1] → MRR = (1.0 + 0.5 + 1.0)/3 = 0.83

**Mean Average Precision (MAP):**
Similar to MRR but accounts for multiple correct answers and gives partial credit:
```
For each query:
  Rank through candidates, track precision as you find correct ones
  Average those precision values
MAP = average across all queries
```
Higher MAP = correct answers appear earlier in rankings

**Top-1 Accuracy:**
Simply: "Pick the highest-scoring candidate per query. How often is it right?"

---

## 🎚️ Threshold Tuning

### The Problem
Models output probabilities (0-1), but we need yes/no decisions.

Default threshold: 0.5
- Probability ≥ 0.5 → "Match"
- Probability < 0.5 → "No match"

But 0.5 might not be optimal!

### The Solution
Try different thresholds (0.1, 0.12, 0.14, ..., 0.9) and measure F1 score for each:

```
Threshold 0.3 → High recall (catch everything) but low precision (many false alarms)
Threshold 0.5 → Balanced
Threshold 0.7 → High precision (few false alarms) but low recall (miss some matches)
```

Pick the threshold with the best F1 (or whatever metric you care about)!

### Coverage vs Accuracy Trade-off
With higher thresholds:
- ✅ Predictions are more accurate
- ❌ You reject more queries (lower coverage)

Example:
- Threshold 0.5: Accept 80% of queries, 85% accurate on accepted
- Threshold 0.7: Accept 50% of queries, 95% accurate on accepted

Choose based on your use case!

---

## 🌟 Summary: The Complete Pipeline Flow

```
1. Load Data
   ↓
2. Clean & Split (80% train, 20% test)
   ↓
3. Baseline (simple average) → benchmark performance
   ↓
4. Scale Features (standardize to mean=0, std=1)
   ↓
5. Train 3 Models with Cross-Validation:
   - Logistic Regression (linear, interpretable)
   - Random Forest (parallel trees, robust)
   - Gradient Boosting (sequential trees, accurate)
   ↓
6. Test Each Model:
   - Classification metrics (AUC, F1, Precision, Recall)
   - Ranking metrics (MAP, MRR, P@1)
   ↓
7. Pick Best Model
   ↓
8. Tune Threshold (optimize F1)
   ↓
9. Apply to New Data with Confidence!
```

Hope this helps! 🚀


