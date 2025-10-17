"""
LLM-based synonym expansion for ICD9 terms.
Generates alternative phrasings and medical synonyms to improve retrieval.
"""
from __future__ import annotations

from typing import List, Optional
from .llm_client import LlmJSONClient, LlmError


SYNONYM_SYSTEM_PROMPT = """You are a medical terminology expert. Given an ICD9 diagnosis code and name, generate alternative medical terms, synonyms, and common phrasings that mean the same thing.

Focus on:
- Medical synonyms (e.g., "venereal disease" → "sexually transmitted disease")
- Common abbreviations (e.g., "STD", "STI")
- Alternative phrasings used in clinical settings
- Both formal and informal terminology

Return up to 6 high-quality alternatives. Only return synonyms that are highly similar in meaning to the original term, Do not include the original term."""


SYNONYM_USER_TEMPLATE = """ICD9 Code: {icd9_code}
ICD9 Name: {icd9_name}

Generate medical synonyms and alternative terms for this diagnosis."""


SYNONYM_SCHEMA = {
    "type": "object",
    "properties": {
        "synonyms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of 4-6 alternative medical terms or synonyms",
            "minItems": 4,
            "maxItems": 8
        }
    },
    "required": ["synonyms"],
    "additionalProperties": False
}


class SynonymResponse:
    """Response from synonym generation LLM."""
    
    def __init__(self, synonyms: List[str]):
        self.synonyms = [s.strip() for s in synonyms if s.strip()]
    
    @classmethod
    def from_dict(cls, data: dict) -> "SynonymResponse":
        return cls(synonyms=data.get("synonyms", []))
    
    def to_dict(self) -> dict:
        return {"synonyms": self.synonyms}


async def generate_synonyms(
    client: LlmJSONClient,
    icd9_code: str,
    icd9_name: str,
    max_attempts: int = 2
) -> List[str]:
    """
    Generate synonyms for an ICD9 term using LLM.
    
    Args:
        client: LLM client for API calls
        icd9_code: ICD9 code (e.g., "099")
        icd9_name: ICD9 diagnosis name (e.g., "Other venereal diseases")
        max_attempts: Maximum retry attempts
    
    Returns:
        List of synonym strings (may be empty if generation fails)
    """
    messages = [
        {"role": "system", "content": SYNONYM_SYSTEM_PROMPT},
        {"role": "user", "content": SYNONYM_USER_TEMPLATE.format(
            icd9_code=icd9_code,
            icd9_name=icd9_name
        )}
    ]
    
    for attempt in range(max_attempts):
        try:
            response, _ = await client.create_and_validate(
                messages,
                SYNONYM_SCHEMA,
                lambda d: SynonymResponse.from_dict(d)
            )
            return response.synonyms
        except LlmError as e:
            if attempt == max_attempts - 1:
                # If all attempts fail, return empty list
                # This allows the pipeline to continue with original term only
                return []
            # Add corrective message and retry
            messages.append({
                "role": "system",
                "content": "Previous response was invalid. Please return a valid JSON with 4-6 medical synonyms."
            })
    
    return []


def build_query_variants(
    icd9_name: str,
    synonyms: List[str],
    include_original: bool = True
) -> List[str]:
    """
    Build list of query strings for multi-query search.
    
    Args:
        icd9_name: Original ICD9 diagnosis name
        synonyms: List of LLM-generated synonyms
        include_original: Whether to include original name as first query
    
    Returns:
        List of query strings to use for retrieval
    """
    queries = []
    
    if include_original:
        queries.append(icd9_name)
    
    # Add each synonym as separate query
    for syn in synonyms:
        if syn and syn.strip():
            queries.append(syn.strip())
    
    return queries

