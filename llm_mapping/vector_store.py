from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from .schemas import Icd10Entry, VectorStoreConfig


try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover - import error clarity
    faiss = None  # type: ignore

_DEFAULT_INDEX_FILENAME = "index.faiss"
_DEFAULT_EMB_FILENAME = "embeddings.npy"
_DEFAULT_META_FILENAME = "metadata.json"
_DEFAULT_CODES_FILENAME = "codes.json"
_DEFAULT_CONFIG_FILENAME = "config.json"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


@dataclass
class _IndexFiles:
    index_path: str
    emb_path: str
    meta_path: str
    codes_path: str
    config_path: str


class Icd10VectorStore:
    """Vector store over the ICD10 universe with FAISS inner-product search."""

    def __init__(self, config: VectorStoreConfig):
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required. Please install faiss-cpu in requirements."
            )
        self.config = config
        self.model = None  # lazy
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        self.code_to_row: Dict[str, int] = {}
        self.row_to_code: List[str] = []
        self.code_to_entry: Dict[str, Icd10Entry] = {}

    # ------------------------ Embedding backend ------------------------
    def _ensure_model(self):
        if self.model is not None:
            return
        if self.config.embedding == "local":
            from sentence_transformers import SentenceTransformer  # lazy import

            self.model = SentenceTransformer(self.config.local_model_name)
        else:
            raise ValueError(
                "OpenAI embeddings backend not supported in offline index build. "
                "Use local backend for building; you may use OpenAI at runtime if desired."
            )

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        assert self.model is not None
        emb = np.array(self.model.encode(texts, normalize_embeddings=False), dtype=np.float32)
        if self.config.normalize_embeddings:
            emb = _normalize_rows(emb)
        return emb

    # ------------------------ Build / Save / Load ------------------------
    def build_from_entries(self, entries: List[Icd10Entry]):
        texts = [e.as_search_text() for e in entries]
        embeddings = self._embed_texts(texts)
        self.embeddings = embeddings
        self.code_to_row = {e.code: i for i, e in enumerate(entries)}
        self.row_to_code = [e.code for e in entries]
        self.code_to_entry = {e.code: e for e in entries}
        self._build_faiss_index(embeddings)

    def build_from_csv(
        self,
        csv_path: str,
        code_col: str = "code",
        name_col: str = "name",
        synonyms_col: Optional[str] = "synonyms",
        description_col: Optional[str] = "description",
    ):
        df = pd.read_csv(csv_path)
        entries: List[Icd10Entry] = []
        for _, row in df.iterrows():
            synonyms: List[str] = []
            if synonyms_col and isinstance(row.get(synonyms_col), str):
                synonyms = [s.strip() for s in row[synonyms_col].split("|") if s.strip()]
            description = None
            if description_col and isinstance(row.get(description_col), str):
                description = row[description_col]
            entries.append(
                Icd10Entry(
                    code=str(row[code_col]).strip(),
                    name=str(row[name_col]).strip(),
                    synonyms=synonyms,
                    description=description,
                )
            )
        self.build_from_entries(entries)

    def _build_faiss_index(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        # Inner product search on normalized vectors approximates cosine similarity
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        self.index = index

    def _files(self, out_dir: str) -> _IndexFiles:
        return _IndexFiles(
            index_path=os.path.join(out_dir, _DEFAULT_INDEX_FILENAME),
            emb_path=os.path.join(out_dir, _DEFAULT_EMB_FILENAME),
            meta_path=os.path.join(out_dir, _DEFAULT_META_FILENAME),
            codes_path=os.path.join(out_dir, _DEFAULT_CODES_FILENAME),
            config_path=os.path.join(out_dir, _DEFAULT_CONFIG_FILENAME),
        )

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        files = self._files(out_dir)
        assert self.index is not None and self.embeddings is not None
        faiss.write_index(self.index, files.index_path)
        np.save(files.emb_path, self.embeddings)
        with open(files.codes_path, "w") as f:
            json.dump(self.row_to_code, f)
        with open(files.meta_path, "w") as f:
            json.dump({c: self.code_to_entry[c].to_dict() for c in self.row_to_code}, f)
        with open(files.config_path, "w") as f:
            json.dump(self.config.to_dict(), f)

    @classmethod
    def load(cls, in_dir: str) -> "Icd10VectorStore":
        files = cls(VectorStoreConfig())._files(in_dir)
        with open(files.config_path) as f:
            cfg = VectorStoreConfig(**json.load(f))
        store = cls(cfg)
        store.index = faiss.read_index(files.index_path)
        store.embeddings = np.load(files.emb_path)
        with open(files.codes_path) as f:
            store.row_to_code = json.load(f)
        store.code_to_row = {c: i for i, c in enumerate(store.row_to_code)}
        with open(files.meta_path) as f:
            meta = json.load(f)
        store.code_to_entry = {c: Icd10Entry(**meta[c]) for c in store.row_to_code}
        return store

    # ------------------------ Query ------------------------
    def exists(self, icd10_code: str) -> bool:
        return icd10_code in self.code_to_row

    def get_entry(self, icd10_code: str) -> Optional[Icd10Entry]:
        return self.code_to_entry.get(icd10_code)

    def search(self, text: str, top_k: int = 50) -> List[Tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")
        q = self._embed_texts([text])
        if self.config.normalize_embeddings:
            q = _normalize_rows(q)
        scores, idx = self.index.search(q, top_k)  # type: ignore
        out: List[Tuple[str, float]] = []
        for j, score in zip(idx[0].tolist(), scores[0].tolist()):
            if j == -1:
                continue
            code = self.row_to_code[j]
            out.append((code, float(score)))
        return out
    
    def search_multi_query(
        self, 
        queries: List[str], 
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Perform multi-query retrieval with max-score merging.
        
        For each query:
        1. Embed the query text
        2. Search the FAISS index for top-k matches
        3. Collect all results
        
        Then merge results by taking the MAXIMUM score for each code across all queries,
        and return the top-k highest scoring codes from the merged results.
        
        Args:
            queries: List of query strings to search for
            top_k: Number of top results to retrieve per query AND return in final merged results
        
        Returns:
            Merged list of (code, max_score) tuples, limited to top_k, sorted by score descending
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")
        
        if not queries:
            return []
        
        # Collect results from each query
        best_scores: Dict[str, float] = {}
        
        for query in queries:
            results = self.search(query, top_k=top_k)
            for code, score in results:
                # Take maximum score across all queries
                if code not in best_scores or score > best_scores[code]:
                    best_scores[code] = score
        
        # Sort by score descending and limit to top_k
        merged = sorted(best_scores.items(), key=lambda kv: kv[1], reverse=True)
        return merged[:top_k]


def _cli_build():
    import argparse
    from .schemas import BuildIndexArgs

    parser = argparse.ArgumentParser(description="Build ICD10 FAISS vector store")
    parser.add_argument("--icd10-csv", required=True, help="CSV with columns: code,name,[synonyms],[description]")
    parser.add_argument("--out-dir", required=True, help="Directory to write the vector store")
    parser.add_argument("--code-col", default="code")
    parser.add_argument("--name-col", default="name")
    parser.add_argument("--synonyms-col", default="synonyms")
    parser.add_argument("--description-col", default="description")
    parser.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    cfg = VectorStoreConfig(embedding="local", local_model_name=args.local_model)
    build_args = BuildIndexArgs(
        icd10_csv=args.icd10_csv,
        out_dir=args.out_dir,
        code_col=args.code_col,
        name_col=args.name_col,
        synonyms_col=args.synonyms_col,
        description_col=args.description_col,
        config=cfg,
    )

    store = Icd10VectorStore(build_args.config)
    store.build_from_csv(
        build_args.icd10_csv,
        code_col=build_args.code_col,
        name_col=build_args.name_col,
        synonyms_col=build_args.synonyms_col,
        description_col=build_args.description_col,
    )
    store.save(build_args.out_dir)
    print(f"Index built and saved to: {build_args.out_dir}")


if __name__ == "__main__":
    _cli_build()
