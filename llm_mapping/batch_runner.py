from __future__ import annotations

import argparse
import asyncio
import csv
import json
from typing import Iterable, List, Optional

import pandas as pd

from .schemas import MappingRecord, DirectCandidate, MappingResult, MappingResultWithAudit
from .vector_store import Icd10VectorStore
from .llm_client import LlmJSONClient
from .mapper import map_one


def _parse_direct_candidates(val: Optional[str]) -> List[DirectCandidate]:
    if not isinstance(val, str) or not val.strip():
        return []
    # Accept formats like: "C83.39|C83.30" or "C83.39:Name A|C83.30:Name B"
    out: List[DirectCandidate] = []
    for part in val.split("|"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            code, name = part.split(":", 1)
            out.append(DirectCandidate(code=code.strip(), name=name.strip()))
        else:
            out.append(DirectCandidate(code=part, name=None))
    return out


def _load_input_csv(path: str) -> List[MappingRecord]:
    df = pd.read_csv(path)
    if not {"icd9_code", "icd9_name"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'icd9_code' and 'icd9_name' columns.")
    direct_col = None
    for cand in ["direct_candidates", "direct", "icd10_direct_candidates"]:
        if cand in df.columns:
            direct_col = cand
            break
    records: List[MappingRecord] = []
    for _, row in df.iterrows():
        cands = _parse_direct_candidates(row[direct_col]) if direct_col else []
        records.append(
            MappingRecord(
                icd9_code=str(row["icd9_code"]).strip(),
                icd9_name=str(row["icd9_name"]).strip(),
                direct_candidates=cands,
            )
        )
    return records


async def _run_batch(
    records: List[MappingRecord],
    store: Icd10VectorStore,
    model: str,
    concurrency: int,
    retrieve_top_k: int,
    max_llm_attempts: int,
    audit: bool,
) -> List[MappingResult]:
    client = LlmJSONClient(model=model)
    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[MappingResult]] = [None] * len(records)

    async def worker(i: int, rec: MappingRecord):
        async with sem:
            res = await map_one(
                rec,
                store,
                client,
                retrieve_top_k=retrieve_top_k,
                max_llm_attempts=max_llm_attempts,
                audit=audit,
            )
            results[i] = res

    await asyncio.gather(*(worker(i, r) for i, r in enumerate(records)))
    return [r for r in results if r is not None]


def _write_results_jsonl(path: str, results: List[MappingResult]):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def _write_results_csv(path: str, results: List[MappingResult]):
    rows = []
    for r in results:
        rows.append(
            {
                "icd9_code": r.input.icd9_code,
                "icd9_name": r.input.icd9_name,
                "selected_code": r.selected_code,
                "selected_name": r.selected_name,
                "confidence": r.confidence,
                "rationale": r.rationale,
                "mapping_category": r.mapping_category,
                "match_specificity": r.match_specificity,
                "external_choice_reason": r.external_choice_reason,
                "chosen_source": r.chosen_source,
                "retrieved_top_k": r.retrieved_top_k,
                "num_attempts": r.num_attempts,
                "attempted_returned_code": getattr(r, "attempted_returned_code", None),
                "attempted_returned_name": getattr(r, "attempted_returned_name", None),
                "salvage_strategy": getattr(r, "salvage_strategy", None),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Batch ICD9→ICD10 mapping with LLM + vector store")
    parser.add_argument("--input", required=True, help="CSV with columns: icd9_code, icd9_name, [direct_candidates]")
    parser.add_argument("--icd10-store", required=True, help="Directory of built ICD10 vector store")
    parser.add_argument("--out-jsonl", default="results.jsonl")
    parser.add_argument("--out-csv", default="results.csv")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retrieve-top-k", type=int, default=40)
    parser.add_argument("--max-llm-attempts", type=int, default=2)
    parser.add_argument("--audit-jsonl", default=None, help="Optional JSONL to write prompt/response audits")
    args = parser.parse_args()

    store = Icd10VectorStore.load(args.icd10_store)
    records = _load_input_csv(args.input)

    results = asyncio.run(
        _run_batch(
            records,
            store,
            model=args.model,
            concurrency=args.concurrency,
            retrieve_top_k=args.retrieve_top_k,
            max_llm_attempts=args.max_llm_attempts,
            audit=args.audit_jsonl is not None,
        )
    )

    _write_results_jsonl(args.out_jsonl, results)
    _write_results_csv(args.out_csv, results)
    print(f"Wrote {len(results)} results to {args.out_jsonl} and {args.out_csv}")

    if args.audit_jsonl:
        with open(args.audit_jsonl, "w") as f:
            for r in results:
                if isinstance(r, MappingResultWithAudit) and r.audit is not None:
                    f.write(json.dumps(r.to_dict()) + "\n")
        print(f"Wrote audits to {args.audit_jsonl}")


if __name__ == "__main__":
    main()


