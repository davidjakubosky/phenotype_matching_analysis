from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

from .schemas import (
    MappingRecord,
    DirectCandidate,
    CandidateOption,
    LlmMappingResponse,
    MappingResult,
    MappingResultWithAudit,
    MappingAudit,
    AttemptAudit,
    PromptMessage,
    MapperConfig,
)
from .vector_store import Icd10VectorStore
from .prompt_builder import build_messages
from .llm_client import LlmJSONClient, LlmError, LlmValidationError
from .synonym_expander import generate_synonyms, build_query_variants


def _merge_candidates(record: MappingRecord, store: Icd10VectorStore, retrieved: List[Tuple[str, float]]) -> List[CandidateOption]:
    # Build direct candidates with verified names from store when possible
    direct: List[CandidateOption] = []
    for d in record.direct_candidates:
        name = d.name
        entry = store.get_entry(d.code)
        if entry is not None:
            # Prefer canonical name
            name = entry.name
        elif name is None:
            name = ""
        direct.append(CandidateOption(code=d.code, name=name, source="direct"))

    # Retrieved
    retrieved_opts: List[CandidateOption] = []
    for code, score in retrieved:
        entry = store.get_entry(code)
        if entry is None:
            continue
        retrieved_opts.append(CandidateOption(code=code, name=entry.name, source="retrieved", score=score))

    # Deduplicate by code, prefer direct info
    seen = {}
    out: List[CandidateOption] = []
    for c in direct + retrieved_opts:
        if c.code in seen:
            continue
        seen[c.code] = True
        out.append(c)
    return out


def _query_text(record: MappingRecord) -> str:
    # Use the ICD9 name; include code as auxiliary text
    return f"{record.icd9_code} | {record.icd9_name}"


async def map_one(
    record: MappingRecord,
    store: Icd10VectorStore,
    client: LlmJSONClient,
    config: Optional[MapperConfig] = None,
    retrieve_top_k: Optional[int] = None,  # deprecated, use config
    max_llm_attempts: Optional[int] = None,  # deprecated, use config
    audit: bool = False,
) -> MappingResult:
    # Handle config with backwards compatibility
    if config is None:
        config = MapperConfig(
            retrieve_top_k=retrieve_top_k or 40,
            max_llm_attempts=max_llm_attempts or 2,
            enable_synonym_expansion=False,
        )
    
    # STEP 1: Retrieve candidates (with optional synonym expansion)
    if config.enable_synonym_expansion:
        # Generate synonyms using LLM
        synonyms = await generate_synonyms(
            client,
            record.icd9_code,
            record.icd9_name
        )
        
        # Build query variants (original + synonyms)
        queries = build_query_variants(record.icd9_name, synonyms, include_original=True)
        
        # Multi-query search with max-score merging
        retrieved = store.search_multi_query(queries, top_k=config.synonym_top_k)
    else:
        # Standard single-query search
        retrieved = store.search(_query_text(record), top_k=config.retrieve_top_k)
    
    allowed = _merge_candidates(record, store, retrieved)

    messages, schema = build_messages(record, allowed)

    last_error: Optional[str] = None
    audit_log: Optional[MappingAudit] = MappingAudit(allowed_candidates=allowed, attempts=[]) if audit else None

    last_attempted_code: Optional[str] = None
    last_attempted_name: Optional[str] = None

    for attempt in range(config.max_llm_attempts):
        try:
            resp, raw_text = await client.create_and_validate(messages, schema, lambda d: LlmMappingResponse(**d))
            if audit_log is not None:
                audit_log.attempts.append(
                    AttemptAudit(
                        attempt_index=attempt,
                        messages=[PromptMessage(**m) for m in messages],
                        json_schema=schema,
                        raw_response_text=raw_text,
                        raw_response_object=resp.to_dict(),
                        attempted_code=resp.best_match_icd10_code,
                        attempted_name=resp.best_match_icd10_name,
                    )
                )
            last_attempted_code = resp.best_match_icd10_code
            last_attempted_name = resp.best_match_icd10_name
        except LlmError as e:
            last_error = str(e)
            # Strengthen instruction and retry
            messages = messages + [
                {
                    "role": "system",
                    "content": (
                        "Previous output was invalid. Choose exactly one ICD10 code from ALLOWED_CANDIDATES only"
                        " or set no_confident_match with null best_match fields. Return strictly valid JSON."
                    ),
                }
            ]
            if audit_log is not None:
                audit_log.attempts.append(
                    AttemptAudit(
                        attempt_index=attempt,
                        messages=[PromptMessage(**m) for m in messages],
                        json_schema=schema,
                        validation_error=last_error,
                        note="Validation error; added corrective system message",
                    )
                )
            continue

        # Validate selection against allowed candidates and vector store
        chosen_code = resp.best_match_icd10_code
        chosen_name: Optional[str] = resp.best_match_icd10_name

        allowed_codes = {c.code for c in allowed}
        code_to_source = {c.code: c.source for c in allowed}
        chosen_source = code_to_source.get(chosen_code) if chosen_code is not None else None
        # Accept DIRECT candidates even if not present in the vector store
        code_in_store = store.exists(chosen_code) if chosen_code is not None else False
        code_valid = (
            chosen_code is not None
            and chosen_code in allowed_codes
            and ((chosen_source == "direct") or code_in_store)
        )

        if not code_valid:
            # If LLM said no confident match, accept as-is
            if resp.confidence == "no_confident_match":
                base = MappingResult(
                    input=record,
                    selected_code=None,
                    selected_name=None,
                    confidence=resp.confidence,
                    rationale=resp.rationale,
                    mapping_category="NONE",
                    match_specificity=resp.match_specificity,
                    external_choice_reason="N/A",
                    chosen_source=None,
                    allowed_candidates=allowed,
                    retrieved_top_k=len(retrieved),
                    validator_notes="No confident match accepted.",
                    num_attempts=attempt + 1,
                    attempted_returned_code=last_attempted_code,
                    attempted_returned_name=last_attempted_name,
                    salvage_strategy=None,
                )
                if audit_log is not None:
                    return MappingResultWithAudit(**base.to_dict(), audit=audit_log)
                return base

            # Else, add a corrective message and retry
            messages = messages + [
                {
                    "role": "system",
                    "content": (
                        "Your selected code was not in ALLOWED_CANDIDATES or is invalid."
                        " You must choose one code from ALLOWED_CANDIDATES only."
                    ),
                }
            ]
            last_error = "Selected invalid code"
            if audit_log is not None:
                audit_log.attempts.append(
                    AttemptAudit(
                        attempt_index=attempt,
                        messages=[PromptMessage(**m) for m in messages],
                        json_schema=schema,
                        validation_error=last_error,
                        note="Invalid selection; retrying with corrective system message",
                    )
                )
            continue

        # Normalize name to canonical store name to prevent hallucinations
        entry = store.get_entry(chosen_code)
        if entry is not None:
            chosen_name = entry.name

        # Determine mapping category (chosen_source already computed)
        if any(dc.code == chosen_code for dc in record.direct_candidates):
            mapping_category = "CLOSE_MATCH"
        else:
            mapping_category = "OTHER_MATCH" if chosen_code is not None else "NONE"

        # Sanitize confidence and rationale if contradictory
        final_confidence = resp.confidence
        final_rationale = resp.rationale
        validator_notes_text = None
        salvage_strategy_text = None
        
        # If we have a valid code but LLM said "no_confident_match", this is contradictory
        # We accept the code but set confidence to "weak" to reflect uncertainty
        if chosen_code is not None and resp.confidence == "no_confident_match":
            final_confidence = "weak"
            salvage_strategy_text = "corrected_confidence_from_no_match_to_weak"
            
            validator_notes_text = (
                f"LLM returned valid code '{chosen_code}' but with contradictory 'no_confident_match' confidence. "
                f"Corrected to 'weak'. Original rationale: '{resp.rationale}'"
            )
            
            # If rationale looks like an error message, replace it with something more informative
            if "invalid" in resp.rationale.lower() or "error" in resp.rationale.lower():
                final_rationale = f"Code selected but with low confidence due to contradictory LLM output"

        # Extract multi-map fields if present
        more_broad_code = getattr(resp, 'more_broad_icd10_code', None)
        more_broad_name = getattr(resp, 'more_broad_icd10_name', None)
        closest_exact_code = getattr(resp, 'closest_exact_icd10_code', None)
        closest_exact_name = getattr(resp, 'closest_exact_icd10_name', None)
        
        # Normalize multi-map names to canonical store names
        if more_broad_code and store.exists(more_broad_code):
            more_broad_entry = store.get_entry(more_broad_code)
            if more_broad_entry:
                more_broad_name = more_broad_entry.name
        if closest_exact_code and store.exists(closest_exact_code):
            closest_exact_entry = store.get_entry(closest_exact_code)
            if closest_exact_entry:
                closest_exact_name = closest_exact_entry.name
        
        base = MappingResult(
            input=record,
            selected_code=chosen_code,
            selected_name=chosen_name,
            confidence=final_confidence,
            rationale=final_rationale,
            mapping_category=mapping_category,
            match_specificity=resp.match_specificity,
            external_choice_reason=resp.external_choice_reason,
            chosen_source=chosen_source, 
            allowed_candidates=allowed,
            retrieved_top_k=len(retrieved),
            validator_notes=validator_notes_text,
            num_attempts=attempt + 1,
            attempted_returned_code=last_attempted_code,
            attempted_returned_name=last_attempted_name,
            salvage_strategy=salvage_strategy_text,
            more_broad_icd10_code=more_broad_code,
            more_broad_icd10_name=more_broad_name,
            closest_exact_icd10_code=closest_exact_code,
            closest_exact_icd10_name=closest_exact_name,
        )
        if audit_log is not None:
            return MappingResultWithAudit(**base.to_dict(), audit=audit_log)
        return base

    # If all attempts fail
    # Salvage fallback: if model kept returning an invalid code, choose a deterministic fallback
    fallback_code: Optional[str] = None
    fallback_name: Optional[str] = None
    if last_attempted_code and any(c.code == last_attempted_code for c in allowed):
        # If the attempted code was in allowed list but failed store.exists (rare), still salvage by allowed
        fallback_code = last_attempted_code
        fallback_name = next((c.name for c in allowed if c.code == last_attempted_code), None)
    elif record.direct_candidates:
        # prefer a direct unspecified/unspecified-side candidate when present
        preferred = [dc for dc in record.direct_candidates if "unspecified" in (dc.name or "").lower()]
        if preferred:
            fallback_code = preferred[0].code
            entry = store.get_entry(fallback_code)
            fallback_name = entry.name if entry else preferred[0].name
        else:
            fallback_code = record.direct_candidates[0].code
            entry = store.get_entry(fallback_code)
            fallback_name = entry.name if entry else record.direct_candidates[0].name
    elif allowed:
        # As a last resort, pick the highest-scoring retrieved candidate
        retrieved_only = [c for c in allowed if c.source == "retrieved"]
        retrieved_only.sort(key=lambda c: (c.score or 0.0), reverse=True)
        if retrieved_only:
            fallback_code = retrieved_only[0].code
            fallback_name = retrieved_only[0].name

    base = MappingResult(
        input=record,
        selected_code=fallback_code,
        selected_name=fallback_name,
        confidence="no_confident_match",
        rationale=(last_error or "Unable to obtain valid selection"),
        mapping_category=("CLOSE_MATCH" if fallback_code and any(dc.code == fallback_code for dc in record.direct_candidates) else ("OTHER_MATCH" if fallback_code else "NONE")),
        match_specificity="MORE_BROAD",
        external_choice_reason="N/A",
        chosen_source=("direct" if fallback_code and any(dc.code == fallback_code for dc in record.direct_candidates) else ("retrieved" if fallback_code else None)),
        allowed_candidates=_merge_candidates(record, store, retrieved),
        retrieved_top_k=len(retrieved),
        validator_notes="LLM attempts exhausted",
        num_attempts=config.max_llm_attempts,
        attempted_returned_code=last_attempted_code,
        attempted_returned_name=last_attempted_name,
        salvage_strategy=(
            "attempted_code" if fallback_code == last_attempted_code else (
                "direct_unspecified" if fallback_code and any("unspecified" in (dc.name or "").lower() and dc.code == fallback_code for dc in record.direct_candidates) else (
                    "direct_first" if fallback_code and any(dc.code == fallback_code for dc in record.direct_candidates) else (
                        "retrieved_top" if fallback_code else "none"
                    )
                )
            )
        ),
    )
    if audit_log is not None:
        return MappingResultWithAudit(**base.to_dict(), audit=audit_log)
    return base


