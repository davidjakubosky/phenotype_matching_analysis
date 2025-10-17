from __future__ import annotations

from typing import Dict, List, Tuple

from .schemas import (
    MappingRecord,
    CandidateOption,
)


def build_json_schema(allowed_codes: Dict) -> Dict:
    # JSON schema enforced by the model when supported; restrict code to allowed set or null
    code_enum = sorted(list(allowed_codes))
    return {
        "name": "icd_mapping_response",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "best_match_icd10_code": {
                    "anyOf": [
                        {"type": "string", "enum": code_enum},
                        {"type": "null"},
                    ]
                },
                "best_match_icd10_name": {"type": ["string", "null"]},
                "confidence": {
                    "type": "string",
                    "enum": ["strong", "medium", "weak", "no_confident_match"],
                },
                "rationale": {"type": "string"},
                "mapping_category": {
                    "type": "string",
                    "enum": ["NONE", "CLOSE_MATCH", "OTHER_MATCH", "MULTI_MAP"],
                },
                "match_specificity": {
                    "type": "string",
                    "enum": ["EXACT", "CLOSE", "MORE_BROAD"],
                },
                "external_choice_reason": {
                    "type": "string",
                    "enum": ["MULTIMAP", "BAD_MAPPING", "N/A"],
                },
                "more_broad_icd10_code": {
                    "anyOf": [
                        {"type": "string", "enum": code_enum},
                        {"type": "null"},
                    ]
                },
                "more_broad_icd10_name": {"type": ["string", "null"]},
                "closest_exact_icd10_code": {
                    "anyOf": [
                        {"type": "string", "enum": code_enum},
                        {"type": "null"},
                    ]
                },
                "closest_exact_icd10_name": {"type": ["string", "null"]},
            },
            "required": [
                "best_match_icd10_code",
                "best_match_icd10_name",
                "confidence",
                "rationale",
                "mapping_category",
                "match_specificity",
                "external_choice_reason",
            ],
            # Keep schema simple to avoid model omissions; enforce remaining rules in instructions/validator
        },
        "strict": True,
    }


def _format_candidates(cands: List[CandidateOption]) -> str:
    # Human-readable + strict list
    lines: List[str] = []
    for c in cands:
        name = c.name.replace("\n", " ")
        lines.append(f"- {c.code}: {name} [source={c.source}{f', score={c.score:.3f}' if c.score is not None else ''}]")
    return "\n".join(lines) if lines else "(none)"


def build_messages(record: MappingRecord, allowed: List[CandidateOption]) -> Tuple[List[Dict], Dict]:
    allowed_codes = {c.code for c in allowed}
    schema = build_json_schema(allowed_codes)

    system = (
        "You are a meticulous medical coding assistant."
        " Your task is to map ICD9 codes to ICD10 codes from the ALLOWED_CANDIDATES list only."
        " Never invent codes or names."
        " If no confident match exists, set best_match_icd10_code and best_match_icd10_name to null and set confidence to 'no_confident_match'."
        "\n\nMatching rules:" 
        "\n1) Prefer a candidate from DIRECT mappings if any are suitable; otherwise choose from retrieved/universe."
        "\n2) If no exact match exists, prefer a more broad term that fully covers the ICD9 concept over picking a too-narrow subset."
        "\n3) **MULTI-MAP DETECTION**: Some ICD9 codes represent COMPOSITE conditions that map to multiple distinct ICD10 codes."
        " Examples: 'Streptococcal sore throat and scarlet fever' (two separate diseases), 'Viral and chlamydial infections' (two organism types),"
        " 'Acute and chronic viral hepatitis' (two temporal categories)."
        "\n3a) If the ICD9 name contains AND/OR between distinct conditions, or spans multiple categories that have separate ICD10 codes, this is a MULTI-MAP situation."
        "\n3b) For MULTI-MAP cases:"
        " - Set mapping_category to 'MULTI_MAP'"
        " - In best_match_icd10_code, select the MORE BROAD code that encompasses all parts (if one exists in candidates)"
        " - Populate more_broad_icd10_code and more_broad_icd10_name with the broader encompassing code"
        " - Populate closest_exact_icd10_code and closest_exact_icd10_name with the most clinically salient specific code"
        " - If no broad encompassing code exists, put the most salient specific code in best_match_icd10_code"
        "\n4) If laterality is not given in the ICD9 input, prefer an 'unspecified side' ICD10 over left/right variants of otherwise identical phrasing."
        "\n5) Do not hallucinate. Only choose codes from ALLOWED_CANDIDATES."
        "\n6) Classify confidence as strong/medium/weak based on textual alignment and inclusivity."
        " If you return a code, confidence must be one of strong/medium/weak. Only use 'no_confident_match' when you return null."
        "\n7) Set mapping_category to:"
        " - 'MULTI_MAP' if ICD9 spans multiple distinct ICD10 concepts"
        " - 'CLOSE_MATCH' if chosen from DIRECT candidates"
        " - 'OTHER_MATCH' if chosen from retrieved/universe"
        " - 'NONE' if nothing was chosen."
        "\n8) Set match_specificity to EXACT/CLOSE/MORE_BROAD based on semantic relation."
        "\n9) If you chose outside DIRECT candidates (and DIRECT exists), set external_choice_reason to:"
        " 'MULTIMAP' if the ICD9 spans multiple specific ICD10s; otherwise 'BAD_MAPPING'."
        " If no DIRECT candidates exist or you selected from DIRECT, set 'N/A'."
        "\n10) IMPORTANT: All code fields must be exactly one of the allowed code strings provided—no variations or partials."
        "\n11) Include all required fields exactly as specified; provide a concise 1-3 sentence rationale explaining your choice and why it's multi-map if applicable."
        "\n\nExample JSON for MULTI-MAP:\n"
        '{"best_match_icd10_code":"B19","best_match_icd10_name":"Unspecified viral hepatitis","confidence":"medium","rationale":"ICD9 070 spans acute, chronic, and unspecified hepatitis; B19 is the broadest unspecified category","mapping_category":"MULTI_MAP","match_specificity":"MORE_BROAD","external_choice_reason":"N/A","more_broad_icd10_code":"B19","more_broad_icd10_name":"Unspecified viral hepatitis","closest_exact_icd10_code":"B17.9","closest_exact_icd10_name":"Acute viral hepatitis, unspecified"}'
        "\n\nReturn ONLY a JSON object matching the provided JSON schema."
    )

    direct_codes = {c.code for c in allowed if c.source == "direct"}
    allowed_text = _format_candidates(allowed)

    user = (
        f"ICD9 Input:\n- Code: {record.icd9_code}\n- Name: {record.icd9_name}\n\n"
        f"ALLOWED_CANDIDATES (choose at most one):\n{allowed_text}\n\n"
        f"DIRECT candidate codes: {sorted(direct_codes) if direct_codes else '[]'}\n"
        "Choose exactly one ICD10 code from ALLOWED_CANDIDATES, or no match."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, schema


