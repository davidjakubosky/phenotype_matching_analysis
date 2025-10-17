python -m llm_mapping.run_from_tsv \
  --icd10-universe-tsv /Users/davidjakubosky/repos/phenotype_matching_analysis/ICD10_mapping_universe.tsv \
  --icd9-input-tsv /Users/davidjakubosky/repos/phenotype_matching_analysis/ICD9_1085_traits_with_candidates.tsv \
  --out-dir /Users/davidjakubosky/projects/llm_pheno_matching/icd9_10_full_run/ \
  --run-name icd9_1085_full_mapping \
  --model gpt-4o-mini \
  --concurrency 12 \
  --retrieve-top-k 40 \
  --max-llm-attempts 2  