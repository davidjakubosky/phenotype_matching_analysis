#!/bin/bash
# Full run on all 1085 ICD9 codes with MULTI-MAP detection
# This script runs the complete ICD9→ICD10 mapping with:
#   - MULTI-MAP detection (is_multi_map field for composite ICD9 codes)
#   - Synonym expansion (improved retrieval for terms like "venereal disease")
#   - Fixed leading zero preservation (e.g., "099" stays as "099")
#   - Top-k limiting in multi-query (prevents excessive candidates)

echo "================================================================================"
echo "FULL RUN: ICD9→ICD10 Mapping with MULTI-MAP Detection + Synonym Expansion"
echo "================================================================================"
echo "Dataset: 1085 ICD9 codes"
echo "Features: MULTI-MAP detection, Synonym expansion, Leading zero preservation"
echo "Output: /Users/davidjakubosky/projects/llm_pheno_matching/icd9_10_full_run_multimap/"
echo ""

python -m llm_mapping.run_from_tsv \
  --icd10-universe-tsv /Users/davidjakubosky/repos/phenotype_matching_analysis/ICD10_mapping_universe.tsv \
  --icd9-input-tsv /Users/davidjakubosky/repos/phenotype_matching_analysis/ICD9_1085_traits_with_candidates.tsv \
  --out-dir /Users/davidjakubosky/projects/llm_pheno_matching/icd9_10_full_run_multimap/ \
  --run-name icd9_1085_multimap_with_synonyms \
  --icd10-store /Users/davidjakubosky/repos/phenotype_matching_analysis/output/icd10_index \
  --model gpt-4o-mini \
  --concurrency 12 \
  --retrieve-top-k 40 \
  --max-llm-attempts 2 \
  --enable-synonym-expansion \
  --synonym-top-k 40

echo ""
echo "================================================================================"
echo "Run complete! Output files:"
echo "  - CSV: icd9_1085_multimap_with_synonyms.csv"
echo "  - JSONL: icd9_1085_multimap_with_synonyms.jsonl"
echo ""
echo "Check for MULTI-MAP cases:"
echo "  python -c \"import pandas as pd; df = pd.read_csv('/Users/davidjakubosky/projects/llm_pheno_matching/icd9_10_full_run_multimap/icd9_1085_multimap_with_synonyms.csv'); multi = df[df['is_multi_map'] == True]; print(f'Found {len(multi)} MULTI-MAP cases'); print(multi[['icd9_code', 'icd9_name', 'more_broad_icd10_code', 'closest_exact_icd10_code']].head(10))\""
echo "================================================================================"

