
"""
Compare results between baseline and synonym expansion tests.
"""
import pandas as pd
import sys

def compare_results(baseline_csv: str, synonym_csv: str):
    """Compare two result files and show differences."""
    
    # Read CSVs with icd9_code as string to preserve leading zeros
    df_baseline = pd.read_csv(baseline_csv, dtype={'icd9_code': str})
    df_synonym = pd.read_csv(synonym_csv, dtype={'icd9_code': str})
    
    print("=" * 80)
    print("COMPARISON: Baseline vs Synonym Expansion")
    print("=" * 80)
    print(f"Baseline file: {baseline_csv}")
    print(f"Synonym file:  {synonym_csv}")
    print(f"Baseline records: {len(df_baseline)}")
    print(f"Synonym records:  {len(df_synonym)}")
    print()
    
    # Clean up codes for joining
    df_baseline['icd9_code'] = df_baseline['icd9_code'].str.strip()
    df_synonym['icd9_code'] = df_synonym['icd9_code'].str.strip()
    
    merged = pd.merge(
        df_baseline,
        df_synonym,
        on='icd9_code',
        suffixes=('_baseline', '_synonym'),
        how='inner'
    )
    
    print(f"Records matched on icd9_code: {len(merged)}")
    print()
    
    # Compare selected codes
    differences = []
    improvements = []
    regressions = []
    
    for _, row in merged.iterrows():
        icd9_code = row['icd9_code']
        icd9_name = row['icd9_name_baseline']
        base_code = row['selected_code_baseline']
        syn_code = row['selected_code_synonym']
        
        if base_code != syn_code:
            differences.append({
                'icd9_code': icd9_code,
                'icd9_name': icd9_name,
                'baseline_code': base_code,
                'baseline_name': row.get('selected_name_baseline', ''),
                'baseline_confidence': row.get('confidence_baseline', ''),
                'synonym_code': syn_code,
                'synonym_name': row.get('selected_name_synonym', ''),
                'synonym_confidence': row.get('confidence_synonym', ''),
            })
            
            # Heuristic: improvement if synonym has higher confidence
            conf_order = {'strong': 3, 'medium': 2, 'weak': 1, 'no_confident_match': 0}
            base_conf_score = conf_order.get(row.get('confidence_baseline', ''), 0)
            syn_conf_score = conf_order.get(row.get('confidence_synonym', ''), 0)
            
            if syn_conf_score > base_conf_score:
                improvements.append(differences[-1])
            elif syn_conf_score < base_conf_score:
                regressions.append(differences[-1])
    
    print(f"Total differences: {len(differences)}/{len(merged)} ({100*len(differences)/len(merged):.1f}%)")
    print(f"  - Potential improvements (higher confidence): {len(improvements)}")
    print(f"  - Potential regressions (lower confidence): {len(regressions)}")
    print(f"  - Same confidence but different code: {len(differences) - len(improvements) - len(regressions)}")
    print()
    
    if differences:
        print("=" * 80)
        print("DETAILED DIFFERENCES")
        print("=" * 80)
        for i, diff in enumerate(differences, 1):
            print(f"\n{i}. ICD9: {diff['icd9_code']} | {diff['icd9_name']}")
            print(f"   Baseline: {diff['baseline_code']} | {diff['baseline_name']}")
            print(f"             Confidence: {diff['baseline_confidence']}")
            print(f"   Synonym:  {diff['synonym_code']} | {diff['synonym_name']}")
            print(f"             Confidence: {diff['synonym_confidence']}")
            
            if diff in improvements:
                print(f"   → IMPROVEMENT ✓")
            elif diff in regressions:
                print(f"   → REGRESSION ✗")
    
    # Check for the specific venereal disease case
    print("\n" + "=" * 80)
    print("SPECIFIC CASE: ICD9 099 (Other venereal diseases)")
    print("=" * 80)
    
    venereal_row = merged[merged['icd9_code'] == '099']
    
    if len(venereal_row) > 0:
        row = venereal_row.iloc[0]
        
        print(f"ICD9: {row['icd9_code']} | {row['icd9_name_baseline']}")
        print(f"  Baseline selected: {row['selected_code_baseline']} | {row.get('selected_name_baseline', '')}")
        print(f"  Baseline confidence: {row.get('confidence_baseline', '')}")
        print(f"  Synonym selected:  {row['selected_code_synonym']} | {row.get('selected_name_synonym', '')}")
        print(f"  Synonym confidence: {row.get('confidence_synonym', '')}")
        
        if row['selected_code_synonym'] == 'A64':
            print(f"  ✓ SUCCESS! Synonym expansion found the correct match (A64)")
        elif row['selected_code_baseline'] == 'A64':
            print(f"  ⚠ Baseline already found A64")
        else:
            print(f"  ✗ Neither found A64 (expected match)")
            print(f"     Expected: A64 | Unspecified sexually transmitted disease")
    else:
        print("  Code 099 not found in test set")
    
    print()
    
    # Summary statistics
    print("=" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)
    print("\nBaseline:")
    print(merged['confidence_baseline'].value_counts().to_string())
    print("\nWith Synonym Expansion:")
    print(merged['confidence_synonym'].value_counts().to_string())
    print()


if __name__ == "__main__":
    baseline = "output/test_comparisons/baseline_40_no_synonyms.csv"
    synonym = "output/test_comparisons/synonym_expansion_40.csv"
    
    if len(sys.argv) > 2:
        baseline = sys.argv[1]
        synonym = sys.argv[2]
    
    compare_results(baseline, synonym)

