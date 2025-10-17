from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .vector_store import Icd10VectorStore, VectorStoreConfig
from .schemas import MappingRecord, DirectCandidate, MappingResultWithAudit, MappingResult
from .mapper import map_one
from .llm_client import LlmJSONClient


def build_store_if_needed(icd10_tsv: str, store_dir: str) -> Icd10VectorStore:
    if os.path.isdir(store_dir) and os.path.exists(os.path.join(store_dir, "index.faiss")):
        return Icd10VectorStore.load(store_dir)

    cfg = VectorStoreConfig(embedding="local", local_model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Icd10VectorStore(cfg)
    # TSV input: icd10_phe, icd10_code
    df = pd.read_csv(icd10_tsv, sep="\t")
    df = df.rename(columns={"icd10_code": "code", "icd10_phe": "name"})
    os.makedirs(store_dir, exist_ok=True)
    tmp_csv = os.path.join(store_dir, "icd10_temp.csv")
    df.to_csv(tmp_csv, index=False)
    store.build_from_csv(tmp_csv, code_col="code", name_col="name", synonyms_col=None, description_col=None)
    store.save(store_dir)
    return store


def parse_potential_matches(cell: Optional[str]) -> List[DirectCandidate]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    out: List[DirectCandidate] = []
    for part in [p.strip() for p in cell.split("|") if p.strip()]:
        code = None
        name = None
        tokens = [t.strip() for t in part.split(":")]
        for i in range(0, len(tokens) - 1, 2):
            key = tokens[i].lower()
            val = tokens[i + 1]
            if key == "icd10_code":
                code = val
            elif key == "phenoname":
                name = val
        if code:
            out.append(DirectCandidate(code=code, name=name))
    return out


def load_input_records(path: str, limit: Optional[int]) -> List[MappingRecord]:
    df = pd.read_csv(path, sep="\t")
    if limit is not None:
        df = df.head(limit)
    recs: List[MappingRecord] = []
    for _, row in df.iterrows():
        recs.append(
            MappingRecord(
                icd9_code=str(row["icd9_code"]).strip(),
                icd9_name=str(row["icd9_phe"]).strip(),
                direct_candidates=parse_potential_matches(row.get("potential_matches")),
            )
        )
    return recs


async def run_all(
    icd10_universe_tsv: str,
    icd9_input_tsv: str,
    out_dir: str,
    run_name: str,
    model: str,
    concurrency: int,
    retrieve_top_k: int,
    max_llm_attempts: int,
    audit_path: Optional[str],
    limit: Optional[int],
    icd10_store: Optional[str],
) -> List[MappingResult]:
    # Prepare store
    store_dir = icd10_store if icd10_store else os.path.join(out_dir, "icd10_index")
    store = build_store_if_needed(icd10_universe_tsv, store_dir)

    # Load input
    records = load_input_records(icd9_input_tsv, limit=limit)
    client = LlmJSONClient(model=model)
    sem = asyncio.Semaphore(concurrency)
    results: List[MappingResult] = []
    do_audit = audit_path is not None

    async def worker(rec: MappingRecord):
        async with sem:
            res = await map_one(
                rec,
                store,
                client,
                retrieve_top_k=retrieve_top_k,
                max_llm_attempts=max_llm_attempts,
                audit=do_audit,
            )
            results.append(res)

    await asyncio.gather(*(worker(r) for r in records))

    # Write outputs
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"{run_name}.jsonl")
    csv_path = os.path.join(out_dir, f"{run_name}.csv")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

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
                "num_attempts": r.num_attempts,
                "attempted_returned_code": getattr(r, "attempted_returned_code", None),
                "attempted_returned_name": getattr(r, "attempted_returned_name", None),
                "salvage_strategy": getattr(r, "salvage_strategy", None),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    if audit_path:
        with open(audit_path, "w") as f:
            for r in results:
                if isinstance(r, MappingResultWithAudit) and r.audit is not None:
                    f.write(json.dumps(r.to_dict()) + "\n")

    print(f"Wrote outputs: {jsonl_path}, {csv_path}{', audits to ' + audit_path if audit_path else ''}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full ICD9→ICD10 mapping from TSVs with configurable output names")
    parser.add_argument("--icd10-universe-tsv", required=True, help="TSV with columns: icd10_phe, icd10_code")
    parser.add_argument("--icd9-input-tsv", required=True, help="TSV with columns: icd9_code, icd9_phe, potential_matches")
    parser.add_argument("--out-dir", default="/tmp/llm_mapping_runs")
    parser.add_argument("--run-name", default=None, help="Base name for outputs; defaults to timestamp")
    parser.add_argument("--icd10-store", default=None, help="Existing ICD10 store directory (skip rebuild)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--retrieve-top-k", type=int, default=40)
    parser.add_argument("--max-llm-attempts", type=int, default=2)
    parser.add_argument("--audit-jsonl", default=None, help="Optional audits JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit (omit to run all)")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    asyncio.run(
        run_all(
            icd10_universe_tsv=args.icd10_universe_tsv,
            icd9_input_tsv=args.icd9_input_tsv,
            out_dir=args.out_dir,
            run_name=run_name,
            model=args.model,
            concurrency=args.concurrency,
            retrieve_top_k=args.retrieve_top_k,
            max_llm_attempts=args.max_llm_attempts,
            audit_path=args.audit_jsonl,
            limit=args.limit,
            icd10_store=args.icd10_store,
        )
    )


if __name__ == "__main__":
    main()


