"""
Simulation tool to test different retrieval strategies for ICD9→ICD10 mapping.
Helps diagnose why certain expected matches are not appearing in top-k results.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple, Optional

from .vector_store import Icd10VectorStore


def format_results(results: List[Tuple[str, float]], store: Icd10VectorStore, limit: int = 20) -> List[str]:
    """Format search results as human-readable lines."""
    lines: List[str] = []
    for i, (code, score) in enumerate(results[:limit], start=1):
        entry = store.get_entry(code)
        name = entry.name if entry else "<missing>"
        lines.append(f"  {i:>3}. {code:<10} score={score:>6.4f}  {name}")
    return lines


def find_rank(results: List[Tuple[str, float]], target_code: str) -> Tuple[int, float]:
    """Find the rank (1-indexed) and score of a target code in results."""
    for i, (code, score) in enumerate(results, start=1):
        if code == target_code:
            return i, score
    return -1, 0.0


def merge_results_max_score(results_list: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    """Merge multiple result lists by taking max score for each code."""
    best: Dict[str, float] = {}
    for results in results_list:
        for code, score in results:
            if code not in best or score > best[code]:
                best[code] = score
    merged = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return merged


class RetrievalStrategy:
    """Base class for a retrieval strategy."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def search(self, store: Icd10VectorStore, icd9_code: str, icd9_name: str, top_k: int) -> List[Tuple[str, float]]:
        """Execute search and return results."""
        raise NotImplementedError


class BaselineStrategy(RetrievalStrategy):
    """Current approach: code | name"""
    
    def __init__(self):
        super().__init__("baseline", "Current approach: 'CODE | NAME'")
    
    def search(self, store: Icd10VectorStore, icd9_code: str, icd9_name: str, top_k: int) -> List[Tuple[str, float]]:
        query = f"{icd9_code} | {icd9_name}"
        return store.search(query, top_k=top_k)


class NameOnlyStrategy(RetrievalStrategy):
    """Name only (no code token)"""
    
    def __init__(self):
        super().__init__("name_only", "Name only (removes code bias)")
    
    def search(self, store: Icd10VectorStore, icd9_code: str, icd9_name: str, top_k: int) -> List[Tuple[str, float]]:
        return store.search(icd9_name, top_k=top_k)


class SynonymExpansionStrategy(RetrievalStrategy):
    """Append synonyms to the query string"""
    
    def __init__(self):
        super().__init__("synonym_expansion", "Name + synonyms concatenated")
        self.synonym_map = {
            "venereal": ["sexually transmitted disease", "sexually transmitted infection", "STD", "STI"],
        }
    
    def search(self, store: Icd10VectorStore, icd9_code: str, icd9_name: str, top_k: int) -> List[Tuple[str, float]]:
        expansions = []
        name_lower = icd9_name.lower()
        for trigger, synonyms in self.synonym_map.items():
            if trigger in name_lower:
                expansions.extend(synonyms)
        
        if expansions:
            query = icd9_name + " | " + " | ".join(expansions)
        else:
            query = icd9_name
        
        return store.search(query, top_k=top_k)


class MultiQueryStrategy(RetrievalStrategy):
    """Run multiple queries and merge by max score"""
    
    def __init__(self):
        super().__init__("multi_query", "Multiple queries merged by max score")
        self.synonym_map = {
            "venereal": ["sexually transmitted disease", "sexually transmitted infection", "STD", "STI", 
                        "unspecified sexually transmitted disease"],
        }
    
    def search(self, store: Icd10VectorStore, icd9_code: str, icd9_name: str, top_k: int) -> List[Tuple[str, float]]:
        queries = [icd9_name]
        name_lower = icd9_name.lower()
        
        for trigger, synonyms in self.synonym_map.items():
            if trigger in name_lower:
                queries.extend(synonyms)
                break
        
        results_list = [store.search(q, top_k=top_k) for q in queries]
        return merge_results_max_score(results_list)


class LargeKStrategy(RetrievalStrategy):
    """Simply retrieve many more candidates"""
    
    def __init__(self, k_multiplier: int = 5):
        super().__init__(f"large_k_{k_multiplier}x", f"Retrieve {k_multiplier}x more candidates")
        self.k_multiplier = k_multiplier
    
    def search(self, store: Icd10VectorStore, icd9_code: str, icd9_name: str, top_k: int) -> List[Tuple[str, float]]:
        return store.search(icd9_name, top_k=top_k * self.k_multiplier)


def run_simulation(
    store_dir: str,
    icd9_code: str,
    icd9_name: str,
    target_icd10: Optional[str] = None,
    top_k: int = 40,
    print_limit: int = 25,
) -> None:
    """
    Run simulation comparing different retrieval strategies.
    
    Args:
        store_dir: Path to the ICD10 vector store directory
        icd9_code: ICD9 code to map
        icd9_name: ICD9 phenotype name
        target_icd10: Expected ICD10 code (to check if it appears in results)
        top_k: Number of candidates to retrieve
        print_limit: Number of top results to print
    """
    print("=" * 80)
    print("ICD9 → ICD10 RETRIEVAL SIMULATION")
    print("=" * 80)
    
    # Load the vector store
    store = Icd10VectorStore.load(store_dir)
    print(f"\nLoaded vector store from: {store_dir}")
    print(f"Store contains {len(store.row_to_code)} ICD10 codes")
    
    # Check if target exists
    if target_icd10:
        target_in_store = store.exists(target_icd10)
        target_entry = store.get_entry(target_icd10)
        target_name = target_entry.name if target_entry else "<missing>"
        print(f"\nTarget code: {target_icd10}")
        print(f"Target in store: {target_in_store}")
        if target_in_store:
            print(f"Target name: {target_name}")
    
    print(f"\nInput:")
    print(f"  ICD9 Code: {icd9_code}")
    print(f"  ICD9 Name: {icd9_name}")
    print(f"  Retrieve top-{top_k} candidates per strategy")
    print()
    
    # Define strategies to test
    strategies: List[RetrievalStrategy] = [
        BaselineStrategy(),
        NameOnlyStrategy(),
        SynonymExpansionStrategy(),
        MultiQueryStrategy(),
        LargeKStrategy(k_multiplier=5),
    ]
    
    # Run each strategy
    all_results = {}
    for strategy in strategies:
        print("=" * 80)
        print(f"STRATEGY: {strategy.name}")
        print(f"  {strategy.description}")
        print("-" * 80)
        
        results = strategy.search(store, icd9_code, icd9_name, top_k)
        all_results[strategy.name] = results
        
        # Check if target appears
        if target_icd10:
            rank, score = find_rank(results, target_icd10)
            if rank > 0:
                print(f"✓ TARGET FOUND: rank={rank}/{len(results)}, score={score:.4f}")
            else:
                print(f"✗ TARGET NOT FOUND in top {len(results)} results")
        
        # Print top results
        print(f"\nTop {min(print_limit, len(results))} results:")
        for line in format_results(results, store, limit=print_limit):
            print(line)
        print()
    
    # Summary comparison
    if target_icd10:
        print("=" * 80)
        print("SUMMARY: Target Code Rankings")
        print("=" * 80)
        for strategy_name, results in all_results.items():
            rank, score = find_rank(results, target_icd10)
            status = f"rank {rank:>3} (score={score:.4f})" if rank > 0 else "NOT FOUND"
            print(f"  {strategy_name:25s}: {status}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate ICD9→ICD10 retrieval with different strategies"
    )
    parser.add_argument(
        "--store-dir",
        default="output/icd10_index",
        help="Path to ICD10 vector store directory"
    )
    parser.add_argument(
        "--icd9-code",
        default="099",
        help="ICD9 code"
    )
    parser.add_argument(
        "--icd9-name",
        default="Other venereal diseases",
        help="ICD9 phenotype name"
    )
    parser.add_argument(
        "--target-icd10",
        default="A64",
        help="Expected ICD10 code to check for"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of candidates to retrieve"
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=25,
        help="Number of top results to print"
    )
    
    args = parser.parse_args()
    
    run_simulation(
        store_dir=args.store_dir,
        icd9_code=args.icd9_code,
        icd9_name=args.icd9_name,
        target_icd10=args.target_icd10,
        top_k=args.top_k,
        print_limit=args.print_limit,
    )


if __name__ == "__main__":
    main()

