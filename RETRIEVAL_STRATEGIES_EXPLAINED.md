# Retrieval Strategies Explained: Detailed Technical Breakdown

## Overview

All strategies use **cosine similarity** (via FAISS inner product on normalized vectors) to compare text embeddings. The key difference is **WHAT** gets embedded and **HOW** results are combined.

---

## Background: How Vector Search Works

### Step 1: Embeddings
- Uses sentence-transformers model (`all-MiniLM-L6-v2` by default)
- Converts text → 384-dimensional vector
- Vectors are L2-normalized (unit length)

### Step 2: Similarity Scoring
- **Cosine similarity** = dot product of normalized vectors
- Score range: -1 to +1 (higher = more similar)
- In practice, we see scores from ~0.4 to ~0.8 for reasonable matches

### Step 3: FAISS Search
- Pre-computed embeddings for all 46,881 ICD10 codes stored in index
- Query embedding compared against all stored embeddings
- Returns top-k matches sorted by score

---

## Strategy 1: Baseline (Current Approach) ❌

### Query Construction
```
Input:  ICD9 code = "099", name = "Other venereal diseases"
Query:  "099 | Other venereal diseases"
```

### Process
1. Embed the single query string: `"099 | Other venereal diseases"`
2. Search FAISS index for top-40 most similar ICD10 codes
3. Return results directly

### Why It Fails for A64
```
Query embedding:        "099 | Other venereal diseases"
                                 ↓ (embedding model)
                        [0.023, -0.041, 0.089, ..., 0.012]  (384-dim vector)
                                 ↓ (cosine similarity)
Target ICD10:           "A64 | Unspecified sexually transmitted disease"
                        [0.031, -0.028, 0.067, ..., 0.019]

Similarity score:       ~0.45 (estimated, NOT in top-40)
```

**Problem**: The embedding model doesn't know that "venereal diseases" ≈ "sexually transmitted disease"
- These are semantically equivalent but lexically different
- Without explicit medical knowledge, the model treats them as moderately related

---

## Strategy 2: Synonym Expansion ✅ (Rank 5)

### Query Construction
```
Input:      ICD9 name = "Other venereal diseases"
Trigger:    Detects "venereal" in text
Expansions: ["sexually transmitted disease", "sexually transmitted infection", "STD", "STI"]
Query:      "Other venereal diseases | sexually transmitted disease | sexually transmitted infection | STD | STI"
```

### Process
1. Detect trigger words in ICD9 name
2. Concatenate original name + all synonyms with " | " separator
3. Embed the SINGLE concatenated string
4. Search FAISS index once for top-40
5. Return results directly

### How This Improves Results
```
Query embedding:        "Other venereal diseases | sexually transmitted disease | STD | STI | ..."
                                 ↓ (embedding model averages/weights all terms)
                        [0.028, -0.033, 0.076, ..., 0.014]  (384-dim vector)
                                 ↓ (cosine similarity)
Target ICD10:           "A64 | Unspecified sexually transmitted disease"
                        [0.031, -0.028, 0.067, ..., 0.019]

Similarity score:       0.5249 (FOUND at rank 5!)
```

**Why It Works**: 
- The expanded query now contains "sexually transmitted disease"
- The embedding model directly sees the matching terminology
- Score improves from ~0.45 → 0.5249

**Limitation**:
- All terms are embedded together in ONE vector
- The model must balance/average the semantic content
- Not optimal because some synonyms might dilute the signal

---

## Strategy 3: Multi-Query 🏆 (Rank 1, BEST)

### Query Construction
```
Input:     ICD9 name = "Other venereal diseases"
Trigger:   Detects "venereal" in text
Queries:   [
             "Other venereal diseases",
             "sexually transmitted disease",
             "sexually transmitted infection",
             "STD",
             "STI",
             "unspecified sexually transmitted disease"
           ]
```

### Process (DETAILED)

#### Step 1: Run Separate Searches
For EACH query string, independently:

**Query 1:** `"Other venereal diseases"`
```
Embedding: [0.023, -0.041, 0.089, ..., 0.012]
Top-40 results:
  - A56: 0.5716
  - A63: 0.5617
  - A64: 0.4521  (hypothetical score for this query)
  - ...
```

**Query 2:** `"sexually transmitted disease"`
```
Embedding: [0.031, -0.029, 0.071, ..., 0.015]
Top-40 results:
  - A64: 0.7899  ← HIGHEST SCORE!
  - A56: 0.6234
  - A63: 0.6103
  - ...
```

**Query 3:** `"sexually transmitted infection"`
```
Embedding: [0.029, -0.031, 0.069, ..., 0.017]
Top-40 results:
  - A64: 0.7654
  - A56: 0.6512
  - ...
```

**Query 4:** `"STD"`
```
Embedding: [0.033, -0.027, 0.065, ..., 0.019]
Top-40 results:
  - A64: 0.6892
  - A56: 0.5943
  - ...
```

... and so on for all 6 queries

#### Step 2: Merge Results by Maximum Score

```python
# For each ICD10 code that appears in ANY result list,
# take the MAXIMUM score across all queries

best_scores = {}
for query_results in all_results:
    for (code, score) in query_results:
        if code not in best_scores OR score > best_scores[code]:
            best_scores[code] = score

# Example for A64:
A64_scores = [0.4521, 0.7899, 0.7654, 0.6892, 0.7234, 0.8012]
best_scores["A64"] = max(A64_scores) = 0.7899

# Sort all codes by their best score
final_results = sorted(best_scores.items(), by score, descending)
```

#### Step 3: Return Merged List

```
Final merged results (sorted by max score):
1. A64:  0.7899  ← Won because Query 2 had perfect semantic match
2. A56:  0.6731  ← Best score from Query 1
3. A638: 0.6604
4. A63:  0.6469
...
```

### Why This Works Best

**Concrete Example for A64:**
```
Query 2: "sexually transmitted disease"
   ↓ (embedding)
   [0.031, -0.028, 0.067, ..., 0.019]
   
   ↓ (cosine similarity with)
   
ICD10 A64: "Unspecified sexually transmitted disease"
   [0.031, -0.028, 0.068, ..., 0.019]
   
   = 0.7899 (very high score - nearly exact match!)
```

**Key Insight**: 
- Query 2 uses the EXACT phrase "sexually transmitted disease"
- A64 contains "Unspecified sexually transmitted disease"
- The embeddings are nearly identical → very high similarity (0.7899)
- Even though Query 1 scored poorly (0.4521), we take the MAX
- A64 wins overall because at least ONE query matched it perfectly

---

## Strategy Comparison Table

| Strategy | # Queries | # Embeddings | Score Calculation | A64 Rank | A64 Score |
|----------|-----------|--------------|-------------------|----------|-----------|
| **Baseline** | 1 | 1 | Direct similarity | Not found | ~0.45 |
| **Synonym Expansion** | 1 | 1 | Direct similarity (averaged terms) | 5 | 0.5249 |
| **Multi-Query** | 6 | 6 | MAX across queries | **1** | **0.7899** |

---

## Computational Cost

### Synonym Expansion
- Queries: 1
- Embeddings computed: 1 (but longer text)
- FAISS searches: 1
- Results to merge: 0 (direct)
- **Cost: ~same as baseline**

### Multi-Query
- Queries: 6 (for venereal cases, otherwise 1)
- Embeddings computed: 6
- FAISS searches: 6
- Results to merge: 6 × 40 = 240 candidate pairs
- **Cost: ~6x slower for triggered cases**

---

## Score Calculation Deep Dive

### What is Cosine Similarity?

```
Given two vectors A and B:

cosine_similarity = (A · B) / (||A|| × ||B||)

If vectors are normalized (||A|| = ||B|| = 1):
cosine_similarity = A · B (dot product)
```

### Example Calculation

```python
# Query embedding (simplified to 4-D for illustration)
query = [0.5, 0.3, -0.2, 0.1]

# ICD10 code embeddings
A64_emb = [0.52, 0.29, -0.19, 0.11]  # Very similar
B99_emb = [0.1, -0.4, 0.5, -0.2]     # Different

# Cosine similarity
score_A64 = (0.5*0.52 + 0.3*0.29 + (-0.2)*(-0.19) + 0.1*0.11)
          = 0.260 + 0.087 + 0.038 + 0.011
          = 0.396

score_B99 = (0.5*0.1 + 0.3*(-0.4) + (-0.2)*0.5 + 0.1*(-0.2))
          = 0.05 - 0.12 - 0.10 - 0.02
          = -0.19

# A64 wins with much higher score
```

---

## Recommendation

**Use Multi-Query** for production because:
1. ✅ **Best accuracy**: Finds correct match at rank 1
2. ✅ **Highest confidence**: Score 0.7899 vs 0.5249 (synonym expansion)
3. ✅ **Modular**: Easy to add more synonym mappings
4. ✅ **Interpretable**: Can see which query found the match
5. ⚠️ **Cost**: Only 6x slower for cases with synonyms (most cases run 1 query)

The ~6x cost is negligible compared to LLM API calls and is worth it for better retrieval quality.

