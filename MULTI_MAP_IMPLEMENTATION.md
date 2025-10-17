# MULTI-MAP Implementation

## Overview

Implemented support for detecting and handling **MULTI-MAP** situations where a single ICD9 code represents multiple distinct ICD10 concepts that cannot be accurately represented by a single code.

---

## Problem Statement

Many ICD9 codes represent composite conditions that should map to multiple ICD10 codes:

### Examples

1. **ICD9 034** | Streptococcal sore throat and scarlet fever
   - Maps to: J02.0 (streptococcal pharyngitis) **AND** A38.x (scarlet fever)
   - Single code loses information

2. **ICD9 070** | Viral hepatitis  
   - Spans: Acute (B15-B17), Chronic (B18.*), Unspecified (B19.*)
   - Single code misclassifies cases

3. **ICD9 078** | Other diseases due to viruses and Chlamydiae
   - Maps to: A74.* (chlamydial) **AND** B33.8/B34.* (viral)
   - Mixed organism types need separate codes

---

## Solution: MULTI-MAP Category

### Schema Changes

#### New Fields in `MappingResult`
```python
mapping_category: Literal["NONE", "CLOSE_MATCH", "OTHER_MATCH", "MULTI_MAP"]
more_broad_icd10_code: Optional[str] = None
more_broad_icd10_name: Optional[str] = None
closest_exact_icd10_code: Optional[str] = None
closest_exact_icd10_name: Optional[str] = None
```

#### Logic
- **`mapping_category = "MULTI_MAP"`**: Indicates composite ICD9
- **`selected_code`**: Contains `more_broad_icd10_code` (the broader encompassing code)
- **`more_broad_icd10_code`**: Broadest code that covers all concepts
- **`closest_exact_icd10_code`**: Most clinically salient specific code

---

## LLM Prompt Instructions

### Detection Criteria

The LLM is instructed to detect MULTI-MAP when:
1. ICD9 name contains "AND" or "OR" between distinct conditions
2. ICD9 spans multiple categories with separate ICD10 codes
3. Examples explicitly mentioned:
   - "Streptococcal sore throat and scarlet fever" (two diseases)
   - "Viral and chlamydial infections" (two organism types)
   - "Acute and chronic viral hepatitis" (two temporal categories)

### LLM Instructions

```
3) **MULTI-MAP DETECTION**: Some ICD9 codes represent COMPOSITE conditions that map to multiple distinct ICD10 codes.

3a) If the ICD9 name contains AND/OR between distinct conditions, or spans multiple categories that have separate ICD10 codes, this is a MULTI-MAP situation.

3b) For MULTI-MAP cases:
 - Set mapping_category to 'MULTI_MAP'
 - In best_match_icd10_code, select the MORE BROAD code that encompasses all parts (if one exists in candidates)
 - Populate more_broad_icd10_code and more_broad_icd10_name with the broader encompassing code
 - Populate closest_exact_icd10_code and closest_exact_icd10_name with the most clinically salient specific code
 - If no broad encompassing code exists, put the most salient specific code in best_match_icd10_code
```

### Example JSON Response

```json
{
  "best_match_icd10_code": "B19",
  "best_match_icd10_name": "Unspecified viral hepatitis",
  "confidence": "medium",
  "rationale": "ICD9 070 spans acute, chronic, and unspecified hepatitis; B19 is the broadest unspecified category",
  "mapping_category": "MULTI_MAP",
  "match_specificity": "MORE_BROAD",
  "external_choice_reason": "N/A",
  "more_broad_icd10_code": "B19",
  "more_broad_icd10_name": "Unspecified viral hepatitis",
  "closest_exact_icd10_code": "B17.9",
  "closest_exact_icd10_name": "Acute viral hepatitis, unspecified"
}
```

---

## CSV Output Format

The CSV now includes additional columns:

| Column | Description |
|--------|-------------|
| `icd9_code` | Input ICD9 code |
| `icd9_name` | Input ICD9 name |
| `selected_code` | Primary ICD10 code (= `more_broad_icd10_code` for MULTI-MAP) |
| `selected_name` | Primary ICD10 name |
| `confidence` | strong/medium/weak/no_confident_match |
| `mapping_category` | NONE/CLOSE_MATCH/OTHER_MATCH/**MULTI_MAP** |
| `more_broad_icd10_code` | **NEW**: Broader encompassing code (MULTI-MAP only) |
| `more_broad_icd10_name` | **NEW**: Name of broader code |
| `closest_exact_icd10_code` | **NEW**: Most specific salient code (MULTI-MAP only) |
| `closest_exact_icd10_name` | **NEW**: Name of specific code |

---

## Implementation Files

### Modified Files

1. **`llm_mapping/schemas.py`**
   - Added `"MULTI_MAP"` to `MappingCategory` literal
   - Added 4 new optional fields to `MappingResult` and `LlmMappingResponse`

2. **`llm_mapping/prompt_builder.py`**
   - Updated JSON schema to include new fields
   - Added comprehensive MULTI-MAP detection instructions
   - Provided example JSON for MULTI-MAP case

3. **`llm_mapping/mapper.py`**
   - Extracts multi-map fields from LLM response using `getattr()`
   - Normalizes multi-map code names via vector store lookup
   - Passes fields through to `MappingResult`

4. **`llm_mapping/run_from_tsv.py`**
   - CSV output includes 4 new columns for multi-map fields
   - Uses `getattr()` for backward compatibility

---

## Usage

### Running with MULTI-MAP Detection

```bash
# Standard run (MULTI-MAP detection enabled automatically)
python -m llm_mapping.run_from_tsv \
  --icd10-universe-tsv ICD10_mapping_universe.tsv \
  --icd9-input-tsv ICD9_1085_traits_with_candidates.tsv \
  --out-dir output/test \
  --run-name test_multimap \
  --limit 40
```

No special flags needed - the LLM will automatically detect and flag MULTI-MAP situations.

### Interpreting Results

Filter for MULTI-MAP cases in output:
```python
import pandas as pd

df = pd.read_csv("output/test_multimap.csv")
multi_maps = df[df['mapping_category'] == 'MULTI_MAP']

for _, row in multi_maps.iterrows():
    print(f"ICD9: {row['icd9_code']} | {row['icd9_name']}")
    print(f"  Broader code: {row['more_broad_icd10_code']} | {row['more_broad_icd10_name']}")
    print(f"  Specific code: {row['closest_exact_icd10_code']} | {row['closest_exact_icd10_name']}")
    print(f"  Rationale: {row['rationale']}")
    print()
```

---

## Expected Test Cases

Based on your examples, these should be detected as MULTI-MAP:

1. **ICD9 034** - Streptococcal sore throat and scarlet fever
2. **ICD9 070** - Viral hepatitis  
3. **ICD9 078** - Other diseases due to viruses and Chlamydiae
4. **ICD9 079.0** - Adenovirus infection (context-dependent: diagnosis vs etiology)
5. **ICD9 079.9** - Unspecified viral and chlamydial infection

---

## Benefits

1. **Transparency**: Explicitly flags composite conditions
2. **Dual Output**: Provides both broad and specific codes
3. **Downstream Flexibility**: Analysts can choose appropriate code based on context
4. **Audit Trail**: Rationale explains why it's a multi-map
5. **Backward Compatible**: Existing code works unchanged; new fields are optional

---

## Future Enhancements

1. **Multiple Exact Codes**: Current schema supports 1 broad + 1 exact; could extend to multiple exact codes
2. **Context Detection**: For cases like 079.0, could detect whether it's diagnosis vs etiology based on surrounding codes
3. **Confidence Per Code**: Separate confidence scores for broad vs exact matches
4. **Direct Map Priority**: If ICD9 has explicit multiple mappings in direct_candidates, could pre-populate multi-map fields

---

## Summary

✅ **MULTI-MAP category added** to schema  
✅ **Dual output fields** (broad + exact) implemented  
✅ **LLM instructions** updated with detection criteria and examples  
✅ **CSV output** includes new columns  
✅ **Zero linting errors**  
✅ **Backward compatible** - existing code unaffected

**Status**: Ready to test on first 40 records to validate detection 🚀

