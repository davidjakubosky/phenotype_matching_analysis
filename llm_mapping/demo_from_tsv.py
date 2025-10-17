from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import List, Optional

import pandas as pd

from .vector_store import Icd10VectorStore, VectorStoreConfig
from .schemas import MappingRecord, DirectCandidate
from .mapper import map_one
from .llm_client import LlmJSONClient


def build_store_if_needed(icd10_tsv: str, out_dir: str) -> Icd10VectorStore:
    if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "index.faiss")):
        return Icd10VectorStore.load(out_dir)

    cfg = VectorStoreConfig(embedding="local", local_model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Icd10VectorStore(cfg)
    # TSV input: icd10_phe, icd10_code
    df = pd.read_csv(icd10_tsv, sep="\t")
    df = df.rename(columns={"icd10_code": "code", "icd10_phe": "name"})
    # Build directly from dataframe
    tmp_csv = os.path.join(out_dir, "icd10_temp.csv")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(tmp_csv, index=False)
    store.build_from_csv(tmp_csv, code_col="code", name_col="name", synonyms_col=None, description_col=None)
    store.save(out_dir)
    return store


def parse_potential_matches(cell: Optional[str]) -> List[DirectCandidate]:
    # Format: ICD10_code:{code}:phenoname:{phenoname} | ICD10_code:{code}:phenoname:{phenoname}
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = [p.strip() for p in cell.split("|") if p.strip()]
    out: List[DirectCandidate] = []
    for p in parts:
        code = None
        name = None
        # tolerate minor spacing/case
        tokens = [t.strip() for t in p.split(":")]
        # Expect tokens like [ICD10_code, CODE, phenoname, NAME]; be defensive
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


def load_input_subset(path: str, limit: int) -> List[MappingRecord]:
    df = pd.read_csv(path, sep="\t")
    recs: List[MappingRecord] = []
    for _, row in df.head(limit).iterrows():
        recs.append(
            MappingRecord(
                icd9_code=str(row["icd9_code"]).strip(),
                icd9_name=str(row["icd9_phe"]).strip(),
                direct_candidates=parse_potential_matches(row.get("potential_matches")),
            )
        )
    return recs


async def run_demo(
    icd10_universe_tsv: str,
    icd9_input_tsv: str,
    out_dir: str,
    limit: int,
    model: str,
    concurrency: int,
    retrieve_top_k: int,
    max_llm_attempts: int,
    audit: bool,
):
    store = build_store_if_needed(icd10_universe_tsv, out_dir=os.path.join(out_dir, "icd10_index"))
    client = LlmJSONClient(model=model)
    records = load_input_subset(icd9_input_tsv, limit=limit)

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def worker(rec: MappingRecord):
        async with sem:
            res = await map_one(rec, store, client, retrieve_top_k=retrieve_top_k, max_llm_attempts=max_llm_attempts, audit=audit)
            results.append(res)

    await asyncio.gather(*(worker(r) for r in records))

    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "demo_results.jsonl")
    csv_path = os.path.join(out_dir, "demo_results.csv")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    # CSV summary
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
    print(f"Wrote demo outputs: {jsonl_path}, {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Demo ICD9→ICD10 mapping from TSV inputs")
    parser.add_argument("--icd10-universe-tsv", required=True, help="TSV with columns: icd10_phe, icd10_code")
    parser.add_argument("--icd9-input-tsv", required=True, help="TSV with columns: icd9_code, icd9_phe, potential_matches")
    parser.add_argument("--out-dir", default="/Users/davidjakubosky/repos/llm_mapping_demo/")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retrieve-top-k", type=int, default=40)
    parser.add_argument("--max-llm-attempts", type=int, default=3)
    parser.add_argument("--audit-jsonl", default=None, help="Optional path to write prompt/response audits")
    args = parser.parse_args()

    results = asyncio.run(
        run_demo(
            icd10_universe_tsv=args.icd10_universe_tsv,
            icd9_input_tsv=args.icd9_input_tsv,
            out_dir=args.out_dir,
            limit=args.limit,
            model=args.model,
            concurrency=args.concurrency,
            retrieve_top_k=args.retrieve_top_k,
            max_llm_attempts=args.max_llm_attempts,
            audit=args.audit_jsonl is not None,
        )
    )

    if args.audit_jsonl:
        with open(args.audit_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"Wrote audits to {args.audit_jsonl}")


if __name__ == "__main__":
    main()


