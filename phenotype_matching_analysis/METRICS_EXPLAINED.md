# Classification Metrics Explained

## 🎯 Understanding the Confusion Matrix

The confusion matrix is the foundation for understanding your model's performance:

```
                    PREDICTED
                 Negative    Positive
ACTUAL  Negative   TN          FP      (True Negatives, False Positives)
        Positive   FN          TP      (False Negatives, True Positives)
```

### What Each Cell Means in Your Context

- **True Negative (TN)**: Correctly predicted that ICD codes DON'T match
- **False Positive (FP)**: Incorrectly said codes match (they don't) ⚠️ FALSE ALARM
- **False Negative (FN)**: Incorrectly said codes don't match (they do) ⚠️ MISSED MATCH
- **True Positive (TP)**: Correctly predicted that ICD codes DO match ✓

---

## 📊 All Metrics Explained

### 1. **Accuracy** - Overall Correctness
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = "What % of all predictions were correct?"
```

**Example:** 85% accuracy means 85 out of 100 predictions were correct.

**⚠️ Warning:** Can be misleading with imbalanced data!
- If 95% of pairs don't match, predicting "no match" for everything gives 95% accuracy!

---

### 2. **Precision** (PPV - Positive Predictive Value)
```
Precision = TP / (TP + FP)
          = "Of all the matches I predicted, how many were actually correct?"
```

**Example:** 80% precision means 80% of your predicted matches are truly matches.

**When to care:** 
- When false positives are costly
- "If I tell a clinician these codes match, how often am I right?"

---

### 3. **Recall** (Sensitivity, TPR - True Positive Rate)
```
Recall = TP / (TP + FN)
       = "Of all actual matches, how many did I find?"
```

**Example:** 70% recall means you found 70% of all true matches (missed 30%).

**When to care:**
- When false negatives are costly
- "Am I missing important matches?"

---

### 4. **Sensitivity** = Recall (Same Thing!)
```
Sensitivity = TP / (TP + FN)
            = True Positive Rate
```

**Medical context:** How good are you at detecting the "disease" (the match)?
- High sensitivity = catches most true matches
- Low sensitivity = misses many true matches

---

### 5. **Specificity** (TNR - True Negative Rate)
```
Specificity = TN / (TN + FP)
            = "Of all actual non-matches, how many did I correctly identify?"
```

**Example:** 90% specificity means you correctly identified 90% of non-matching pairs.

**Medical context:** How good are you at ruling out "non-disease" (non-match)?
- High specificity = rarely cry wolf (few false alarms)
- Low specificity = many false alarms

---

### 6. **F1 Score** - Balanced Metric
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonic mean of Precision and Recall
```

**Why harmonic mean?** Punishes extreme imbalances.
- If Precision = 100% but Recall = 10%, F1 = 18% (not 55%)
- Forces you to balance both metrics

**When to use:** When you care equally about precision and recall.

---

### 7. **AUC (Area Under ROC Curve)**
```
AUC = Area under curve of TPR vs FPR across all thresholds
    = "How well can the model separate classes?"
```

**Interpretation:**
- 0.5 = Random guessing (coin flip)
- 0.7-0.8 = Acceptable
- 0.8-0.9 = Good
- 0.9+ = Excellent

**Advantage:** Threshold-independent! Shows model's inherent ability to discriminate.

---

### 8. **NPV (Negative Predictive Value)**
```
NPV = TN / (TN + FN)
    = "Of all the non-matches I predicted, how many were actually non-matches?"
```

**Example:** 95% NPV means when you say "no match", you're right 95% of the time.

**When to care:** When you trust the model to rule things out.

---

## 🎭 The Sensitivity-Specificity Trade-off

**They're inversely related!** Moving the threshold changes both:

```
Lower Threshold (e.g., 0.3):
  ✅ Higher Sensitivity (catch more matches)
  ❌ Lower Specificity (more false alarms)
  → "When in doubt, predict match"

Higher Threshold (e.g., 0.7):
  ❌ Lower Sensitivity (miss some matches)
  ✅ Higher Specificity (fewer false alarms)
  → "Only predict match when very confident"
```

### Visual Example:
```
Threshold 0.3:  Sensitivity = 95%, Specificity = 60%
                → Cast a wide net, catch almost everything
                
Threshold 0.5:  Sensitivity = 85%, Specificity = 85%
                → Balanced
                
Threshold 0.7:  Sensitivity = 60%, Specificity = 95%
                → Be very selective, high confidence only
```

---

## 🏥 Medical Analogy

Think of this as a disease screening test:

| Metric | Question | Example |
|--------|----------|---------|
| **Sensitivity** | "Of all sick patients, how many did the test catch?" | Cancer screening: want HIGH sensitivity (don't miss cancer) |
| **Specificity** | "Of all healthy patients, how many did the test correctly say were healthy?" | Pregnancy test: want HIGH specificity (don't falsely alarm) |
| **Precision** | "Of all positive test results, how many are truly sick?" | Follow-up test: want HIGH precision (don't waste resources on false positives) |
| **NPV** | "If test is negative, how confident am I the patient is healthy?" | Rule-out test: want HIGH NPV (trust negative results) |

---

## 🎯 Which Metrics to Prioritize for ICD Code Matching?

### **Scenario 1: Initial Candidate Generation**
**Goal:** Don't miss potential matches (cast wide net)
- **Prioritize:** High Sensitivity/Recall, MAP, MRR
- **Accept:** Lower specificity (more false positives OK)
- **Reason:** Human will review candidates anyway

### **Scenario 2: Automated Matching (No Review)**
**Goal:** Only auto-match when very confident
- **Prioritize:** High Precision, High Specificity
- **Accept:** Lower recall (will manually handle uncertain cases)
- **Reason:** False positives cause problems downstream

### **Scenario 3: Balanced Production System**
**Goal:** Good all-around performance
- **Prioritize:** F1, AUC, Top-1 Accuracy
- **Tune threshold** to achieve desired Sensitivity/Specificity trade-off
- **Reason:** Need to balance all error types

---

## 📈 Interpreting Your Results

### Good Signs ✅
- **High AUC (>0.85):** Model can distinguish matches from non-matches
- **High Top-1 Accuracy (>0.80):** Correct answer is usually ranked #1
- **High Specificity (>0.90):** Not many false alarms
- **Balanced Precision/Recall:** Not sacrificing one for the other

### Warning Signs ⚠️
- **High accuracy but low F1:** Likely due to class imbalance
- **High precision but low recall:** Missing many true matches
- **High recall but low precision:** Too many false positives
- **Low specificity:** Predicting "match" too liberally

### Example Interpretation:
```
Model: Random Forest
  AUC: 0.92           ✅ Excellent separation ability
  Accuracy: 0.88      ✅ Good overall
  F1: 0.75            ✅ Decent balance
  Sensitivity: 0.82   ✅ Catches most matches
  Specificity: 0.90   ✅ Few false alarms
  Top-1 Accuracy: 0.85 ✅ Ranking works well

→ This is a strong model! 
→ Consider threshold tuning to push sensitivity higher if needed.
```

---

## 🔧 Practical Tips

### 1. **Always Look Beyond Accuracy**
With 10:1 negative:positive ratio, accuracy is misleading!

### 2. **Use Confusion Matrix First**
Understand where errors happen before diving into metrics.

### 3. **Threshold Tuning is Key**
Default 0.5 is arbitrary! Tune to your use case.

### 4. **Consider Costs**
- What's worse: missing a match (FN) or false alarm (FP)?
- Medical codes: FN might mean incorrect billing
- Medical codes: FP might mean invalid clinical decision

### 5. **Ranking Metrics for Search**
For "find best match per query" tasks:
- **MAP, MRR, P@1** are more relevant than precision/recall
- **Top-1 Accuracy** is your north star metric

---

## 📚 Quick Reference Table

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **Accuracy** | (TP+TN)/(All) | [0,1] | Overall correctness |
| **Precision** | TP/(TP+FP) | [0,1] | Of predicted positives, % correct |
| **Recall** | TP/(TP+FN) | [0,1] | Of actual positives, % found |
| **Sensitivity** | = Recall | [0,1] | True Positive Rate |
| **Specificity** | TN/(TN+FP) | [0,1] | True Negative Rate |
| **F1** | 2PR/(P+R) | [0,1] | Harmonic mean of P and R |
| **AUC** | ∫ROC | [0,1] | Discrimination ability |
| **PPV** | = Precision | [0,1] | Positive Predictive Value |
| **NPV** | TN/(TN+FN) | [0,1] | Negative Predictive Value |

---

## 🎓 Summary

**For your ICD code matching project:**
1. ✅ **Added:** Sensitivity, Specificity, PPV, NPV, Confusion Matrix
2. 🎯 **Focus on:** AUC, F1, Top-1 Accuracy, Sensitivity/Specificity trade-off
3. 🔧 **Next step:** Tune threshold based on your use case needs
4. 📊 **Monitor:** Confusion matrix to understand error patterns

The notebook now gives you a complete picture of model performance! 🚀


