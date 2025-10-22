#!/bin/bash
# Test script: First 40 records WITHOUT synonym expansion (baseline)

echo "====================================================================="
echo "TEST 1: Baseline (NO synonym expansion) - First 40 records"
echo "====================================================================="

python -m llm_mapping.run_from_tsv \
  --icd10-universe-tsv /Users/davidjakubosky/repos/phenotype_matching_analysis/ICD10_mapping_universe.tsv \
  --icd9-input-tsv /Users/davidjakubosky/repos/phenotype_matching_analysis/ICD9_1085_traits_with_candidates.tsv \
  --out-dir /Users/davidjakubosky/repos/phenotype_matching_analysis/output/test_comparisons \
  --run-name baseline_40_no_synonyms \
  --icd10-store /Users/davidjakubosky/repos/phenotype_matching_analysis/output/icd10_index \
  --model gpt-4o-mini \
  --concurrency 12 \
  --retrieve-top-k 40 \
  --max-llm-attempts 2 \
  --limit 200

echo ""
echo "Baseline test complete!"
echo "Output: output/test_comparisons/baseline_40_no_synonyms.csv"

