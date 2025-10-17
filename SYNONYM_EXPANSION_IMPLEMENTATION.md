# Synonym Expansion Implementation

## Overview

This document describes the implementation of **LLM-based synonym expansion** for improving ICD9→ICD10 code retrieval accuracy.

## Problem Statement

The baseline retrieval approach uses cosine similarity between ICD9 term embeddings and ICD10 term embeddings. However, this fails when semantically equivalent terms use different vocabulary:

- **Example**: ICD9 "099 - Other venereal diseases" 
- **Expected match**: ICD10 "A64 - Unspecified sexually transmitted disease"
- **Problem**: The terms "venereal diseases" and "sexually transmitted disease" are synonyms, but the embedding model doesn't capture this well enough, resulting in A64 not appearing in top-40 candidates.

## Solution: Multi-Query Retrieval with LLM-Generated Synonyms

### Architecture

```
ICD9 Term → LLM Synonym Generator → Multiple Query Strings
                                            ↓
                                    Multi-Query Search
                                            ↓
                                    Max-Score Merging
                                            ↓
                                    Top-K Candidates → Final LLM Selection
```

### Implementation Steps

#### 1. **LLM Synonym Generation** (`llm_mapping/synonym_expander.py`)
- Uses LLM to generate 4-6 medical synonyms for each ICD9 term
- Focuses on alternative terminology, abbreviations, and clinical phrasings
- Returns empty list on failure (graceful degradation to baseline)

#### 2. **Multi-Query Search** (`llm_mapping/vector_store.py`)
- New method: `Icd10VectorStore.search_multi_query(queries, top_k)`
- Runs separate embedding + FAISS search for each query
- Merges results by taking **maximum score** for each ICD10 code
- Re-sorts by merged scores

#### 3. **Mapper Integration** (`llm_mapping/mapper.py`)
- Updated `map_one()` to accept `MapperConfig`
- When `config.enable_synonym_expansion = True`:
  1. Generate synonyms via LLM
  2. Build query variants (original + synonyms)
  3. Use multi-query search
- Otherwise: use baseline single-query search

#### 4. **Configuration** (`llm_mapping/schemas.py`)
- New `MapperConfig` dataclass:
  - `retrieve_top_k`: Standard retrieval limit (default: 40)
  - `max_llm_attempts`: Max LLM retry attempts (default: 2)
  - `enable_synonym_expansion`: Enable/disable feature (default: False)
  - `synonym_top_k`: Top-K per query in multi-query mode (default: 40)

#### 5. **CLI Support** (`llm_mapping/run_from_tsv.py`)
- New flags:
  - `--enable-synonym-expansion`: Turn on the feature
  - `--synonym-top-k N`: Control retrieval size per query

## How It Works: Detailed Example

### Input
```
ICD9: 099 | Other venereal diseases
```

### Step 1: LLM Generates Synonyms
```
Synonyms:
1. "sexually transmitted disease"
2. "sexually transmitted infection"
3. "STD"
4. "STI"
5. "venereal infection"
6. "unspecified sexually transmitted disease"
```

### Step 2: Multi-Query Retrieval

For EACH query string, we:
1. Embed the text → 384-dim vector
2. Search FAISS index → top-40 ICD10 codes with scores
3. Collect results

```
Query 1: "Other venereal diseases"
  Results: [(A56, 0.57), (A63, 0.56), (A64, 0.48), ...]

Query 2: "sexually transmitted disease"
  Results: [(A64, 0.79), (A56, 0.62), (A63, 0.60), ...]  ← A64 ranks #1!

Query 3: "sexually transmitted infection"
  Results: [(A64, 0.77), (A56, 0.65), ...]

... (more queries)
```

### Step 3: Merge by Maximum Score

```python
For each ICD10 code across all query results:
  final_score[code] = max(score_query1, score_query2, ..., score_queryN)

Example:
  A64 scores: [0.48, 0.79, 0.77, 0.69, 0.72, 0.80]
  A64 final: max() = 0.80 ← Wins because Query 2 matched perfectly!
```

### Step 4: Pass Top Candidates to LLM

The merged top-K candidates (with A64 now ranked highly) are passed to the final LLM for selection.

## Key Advantages

1. **Semantic Flexibility**: Captures synonyms the embedding model misses
2. **Robust**: If any synonym matches well, the target code is retrieved
3. **Backward Compatible**: Disabled by default, no breaking changes
4. **Graceful Degradation**: If synonym generation fails, falls back to baseline

## Cost Considerations

### Additional LLM Calls
- **Per record**: 1 extra LLM call for synonym generation
- **Cost**: ~4K input tokens + ~200 output tokens per synonym generation
- **With 40 records**: ~40 extra LLM calls

### Additional Embeddings
- **Per synonym query**: 1 embedding + 1 FAISS search
- **Typical**: 5-6 queries per record (1 original + 4-5 synonyms)
- **Cost**: 5-6x embedding/search overhead (still cheap compared to LLM)

### Total Impact
For 40 records with synonym expansion:
- **Baseline**: 40 LLM selection calls
- **With synonyms**: 40 synonym generation + 40 selection = 80 LLM calls
- **Cost increase**: ~2x LLM calls, but improved accuracy

## Usage

### Test Scripts

Run baseline test (no synonym expansion):
```bash
./test_baseline_40.sh
```

Run with synonym expansion:
```bash
./test_synonym_expansion_40.sh
```

Compare results:
```bash
python compare_results.py
```

### Manual Usage

```bash
python -m llm_mapping.run_from_tsv \
  --icd10-universe-tsv ICD10_mapping_universe.tsv \
  --icd9-input-tsv ICD9_1085_traits_with_candidates.tsv \
  --out-dir output/test \
  --run-name test_with_synonyms \
  --enable-synonym-expansion \
  --synonym-top-k 40 \
  --limit 40
```

### In Code

```python
from llm_mapping.mapper import map_one
from llm_mapping.schemas import MapperConfig

config = MapperConfig(
    retrieve_top_k=40,
    max_llm_attempts=2,
    enable_synonym_expansion=True,  # Enable feature
    synonym_top_k=40
)

result = await map_one(record, store, client, config=config)
```

## Files Changed

### New Files
- `llm_mapping/synonym_expander.py` - LLM synonym generation
- `test_baseline_40.sh` - Baseline test script
- `test_synonym_expansion_40.sh` - Synonym expansion test script
- `compare_results.py` - Result comparison utility

### Modified Files
- `llm_mapping/schemas.py` - Added `MapperConfig`
- `llm_mapping/vector_store.py` - Added `search_multi_query()`
- `llm_mapping/mapper.py` - Integrated synonym expansion
- `llm_mapping/run_from_tsv.py` - Added CLI flags

### Documentation
- `RETRIEVAL_STRATEGIES_EXPLAINED.md` - Technical deep dive
- `SYNONYM_EXPANSION_IMPLEMENTATION.md` - This file
- `multi_query_example.txt` - Step-by-step example

## Testing

To verify the implementation works correctly:

1. **Run baseline**: `./test_baseline_40.sh`
2. **Run with synonyms**: `./test_synonym_expansion_40.sh`
3. **Compare**: `python compare_results.py`

Key metrics to check:
- Does ICD9 "099" now map to A64 with synonym expansion?
- How many mappings changed?
- Did confidence scores improve?

## Next Steps

1. Run tests on first 40 records
2. Analyze results and accuracy improvement
3. If successful, consider:
   - Expanding to full dataset
   - Tuning synonym generation prompt
   - Adding domain-specific synonym rules
   - Caching synonym results to reduce LLM calls

