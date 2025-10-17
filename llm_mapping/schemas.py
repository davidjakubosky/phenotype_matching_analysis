from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict, Any


@dataclass
class Icd10Entry:
    code: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def as_search_text(self) -> str:
        parts: List[str] = [self.code, self.name]
        if self.synonyms:
            parts.extend(self.synonyms)
        if self.description:
            parts.append(self.description)
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DirectCandidate:
    code: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingRecord:
    icd9_code: str
    icd9_name: str
    direct_candidates: List[DirectCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return result


Confidence = Literal["strong", "medium", "weak", "no_confident_match"]
MappingCategory = Literal["NONE", "CLOSE_MATCH", "OTHER_MATCH", "MULTI_MAP"]
Specificity = Literal["EXACT", "CLOSE", "MORE_BROAD"]
ExternalChoiceReason = Literal["MULTIMAP", "BAD_MAPPING", "N/A"]


@dataclass
class LlmMappingResponse:
    best_match_icd10_code: Optional[str]
    best_match_icd10_name: Optional[str]
    confidence: Confidence
    rationale: str
    mapping_category: MappingCategory
    match_specificity: Specificity
    external_choice_reason: ExternalChoiceReason
    # Multi-map support: populated only when mapping_category is "MULTI_MAP"
    more_broad_icd10_code: Optional[str] = None
    more_broad_icd10_name: Optional[str] = None
    closest_exact_icd10_code: Optional[str] = None
    closest_exact_icd10_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateOption:
    code: str
    name: str
    source: Literal["direct", "retrieved"]
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingResult:
    input: MappingRecord
    selected_code: Optional[str]
    selected_name: Optional[str]
    confidence: Confidence
    rationale: str
    mapping_category: MappingCategory
    match_specificity: Specificity
    external_choice_reason: ExternalChoiceReason
    chosen_source: Optional[Literal["direct", "retrieved"]] = None
    allowed_candidates: List[CandidateOption] = field(default_factory=list)
    retrieved_top_k: int = 0
    validator_notes: Optional[str] = None
    num_attempts: int = 0
    # Track what LLM originally returned (may differ from selected_code after validation)
    attempted_returned_code: Optional[str] = None
    attempted_returned_name: Optional[str] = None
    salvage_strategy: Optional[str] = None  # e.g., "corrected_confidence", "accepted_valid_code"
    # Multi-map support: when ICD9 maps to multiple ICD10 codes
    more_broad_icd10_code: Optional[str] = None
    more_broad_icd10_name: Optional[str] = None
    closest_exact_icd10_code: Optional[str] = None
    closest_exact_icd10_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return result


# -------------------- Prompt auditing (optional) --------------------
@dataclass
class PromptMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttemptAudit:
    attempt_index: int
    messages: List[PromptMessage]
    json_schema: Optional[Dict] = None
    raw_response_text: Optional[str] = None
    raw_response_object: Optional[Dict] = None
    validation_error: Optional[str] = None
    note: Optional[str] = None
    attempted_code: Optional[str] = None
    attempted_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingAudit:
    allowed_candidates: List[CandidateOption] = field(default_factory=list)
    attempts: List[AttemptAudit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Attach audit to results when requested
@dataclass
class MappingResultWithAudit(MappingResult):
    audit: Optional[MappingAudit] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return result


@dataclass
class VectorStoreConfig:
    backend: Literal["faiss"] = "faiss"
    embedding: Literal["local", "openai"] = "local"
    local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    normalize_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MapperConfig:
    """Configuration for the ICD9→ICD10 mapping process."""
    retrieve_top_k: int = 40
    max_llm_attempts: int = 2
    enable_synonym_expansion: bool = False
    synonym_top_k: int = 40  # top_k per query when using synonym expansion
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuildIndexArgs:
    icd10_csv: str
    out_dir: str
    code_col: str = "code"
    name_col: str = "name"
    synonyms_col: Optional[str] = "synonyms"
    description_col: Optional[str] = "description"
    config: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
