from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

from .vector_store import Icd10VectorStore


def _format_results(results: List[Tuple[str, float]], store: Icd10VectorStore, limit: int = 20) -> List[str]:
    lines: List[str] = []
    for i, (code, score) in enumerate(results[:limit], start=1):
        entry = store.get_entry(code)
        name = entry.name if entry else "<missing>"
        lines.append(f"{i:>2}. {code:<8} {score:>7.4f}  {name}")
    return lines


def _find_rank(results: List[Tuple[str, float]], target_code: str) -> Tuple[int, float]:
    for i, (code, score) in enumerate(results, start=1):
        if code == target_code:
            return i, score
    return -1, 0.0


def _merge_results_max(results_list: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for results in results_list:
        for code, score in results:
            if code not in best or score > best[code]:
                best[code] = score
    merged = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return merged


def build_queries(icd9_code: str, icd9_name: str) -> Dict[str, List[str]]:
    # Baseline replicates current logic including the ICD9 code token
    baseline = [f"{icd9_code} | {icd9_name}"]

    # Name-only removes the code token which can bias towards x99 ICD10s
    name_only = [icd9_name]

    # Lightweight synonym expansion for venereal/STD/STI terms to bridge to A64 wording
    expansions: List[str] = []
    lname = icd9_name.lower()
    if ("venereal" in lname) or ("std" in lname) or ("sexually transmitted" in lname) or ("sti" in lname):
        expansions.extend(
            [
                "sexually transmitted disease",
                "sexually transmitted infection",
                "STD",
                "STI",
                "venereal disease",
                "unspecified sexually transmitted disease",
            ]
        )

    expanded_single = [icd9_name + (" | " + " | ".join(expansions) if expansions else "")]

    # Multi-query variant: query each expansion separately and merge by max score
    multi_queries = [icd9_name] + expansions if expansions else [icd9_name]

    return {
        "baseline": baseline,
        "name_only": name_only,
        "expanded_concat": expanded_single,
        "expanded_multi": multi_queries,
    }


def run_simulation(
    store_dir: str,
    icd9_code: str,
    icd9_name: str,
    target_icd10: str = "A64",
    top_k_small: int = 40,
    top_k_large: int = 200,
    print_limit: int = 20,
) -> None:
    store = Icd10VectorStore.load(store_dir)
    target_in_store = store.exists(target_icd10)
    target_entry = store.get_entry(target_icd10)
    target_name = target_entry.name if target_entry else "<missing>"

    print(f"Loaded store: {store_dir}")
    print(f"Target {target_icd10} present: {target_in_store} -> {target_name}")
    print()

    queries = build_queries(icd9_code, icd9_name)

    def search_set(qs: List[str], k: int) -> List[Tuple[str, float]]:
        if len(qs) == 1:
            return store.search(qs[0], top_k=k)
        # Multi-query: merge by max
        results_list = [store.search(q, top_k=k) for q in qs]
        return _merge_results_max(results_list)

    # Evaluate with small and large candidate lists
    for k in (top_k_small, top_k_large):
        print(f"=== Retrieval k={k} ===")
        for variant_name, qs in queries.items():
            results = search_set(qs, k)
            rank, score = _find_rank(results, target_icd10)
            print(f"-- {variant_name} --  A64 rank: {rank if rank != -1 else 'not_in_top_k'}  score: {score:.4f}")
            for line in _format_results(results, store, limit=print_limit):
                print(line)
            print()


def main():
    parser = argparse.ArgumentParser(description="Simulate ICD9→ICD10 retrieval variants for a single example")
    parser.add_argument("--store-dir", required=True, help="Path to built ICD10 vector store (directory)")
    parser.add_argument("--icd9-code", default="099", help="ICD9 code text")
    parser.add_argument("--icd9-name", default="Other venereal diseases", help="ICD9 name/phenotype")
    parser.add_argument("--target-icd10", default="A64", help="Expected ICD10 code to check rank for")
    parser.add_argument("--top-k-small", type=int, default=40)
    parser.add_argument("--top-k-large", type=int, default=200)
    parser.add_argument("--print-limit", type=int, default=20)
    args = parser.parse_args()

    run_simulation(
        store_dir=args.store_dir,
        icd9_code=args.icd9_code,
        icd9_name=args.icd9_name,
        target_icd10=args.target_icd10,
        top_k_small=args.top_k_small,
        top_k_large=args.top_k_large,
        print_limit=args.print_limit,
    )


if __name__ == "__main__":
    main()


