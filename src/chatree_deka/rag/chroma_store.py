from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import pyarrow.dataset as ds
import requests
from pythainlp.util import normalize


LAW_PARQUET_PATH = (
    Path(__file__).resolve().parent.parent
    / "law"
    / "ccl-00000-of-00001.parquet"
)

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"

COLLECTION_NAME = "thai_civil_law"

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

BATCH_SIZE = 64


class OllamaEmbeddingFunction:
    """Embedding function using local Ollama."""

    def __init__(
        self,
        model_name: str = OLLAMA_EMBEDDING_MODEL,
        url: str = OLLAMA_EMBED_URL,
    ):
        self.model_name = model_name
        self.url = url

    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "input": input,
                },
                timeout=120,
            )

            response.raise_for_status()
            data = response.json()

            if "embeddings" in data:
                return data["embeddings"]

            if "embedding" in data:
                return [data["embedding"]]

            raise RuntimeError(f"Unexpected Ollama response: {data}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to get embeddings from Ollama "
                f"using model '{self.model_name}'"
            ) from e

    def name(self) -> str:
        return f"ollama-{self.model_name}"


def _query_text(text: str) -> str:
    return f"query: {normalize(text.strip())}"


def _passage_text(text: str) -> str:
    return f"passage: {normalize(text.strip())}"


@lru_cache(maxsize=1)
def _embedding_function() -> OllamaEmbeddingFunction:
    return OllamaEmbeddingFunction()


@lru_cache(maxsize=1)
def _client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection() -> chromadb.Collection:
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _read_rows(limit: int | None = None):
    dataset = ds.dataset(LAW_PARQUET_PATH)

    yielded = 0

    for batch in dataset.to_batches(batch_size=256):
        rows = batch.to_pylist()

        if limit is not None:
            remaining = limit - yielded

            if remaining <= 0:
                break

            rows = rows[:remaining]

        yielded += len(rows)

        yield rows


def _iter_law_entries(row: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    for field_name in ("relevant_laws", "reference_laws"):
        for item in row.get(field_name, []) or []:
            entries.append(
                {
                    "law_name": normalize(
                        str(item.get("law_name", "")).strip()
                    ),
                    "section_num": normalize(
                        str(item.get("section_num", "")).strip()
                    ),
                    "section_content": normalize(
                        str(item.get("section_content", "")).strip()
                    ),
                }
            )

    return entries


def _build_document_text(entry: dict[str, str]) -> str:
    parts = []

    law_name = entry.get("law_name", "")
    section_num = entry.get("section_num", "")
    section_content = entry.get("section_content", "")

    if law_name:
        parts.append(f"กฎหมาย: {law_name}")

    if section_num:
        parts.append(f"มาตรา {section_num}")

    if section_content:
        parts.append(section_content)

    return "\n".join(parts).strip()


def _add_batch(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, str]],
):
    if not documents:
        return

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


def build_index(
    force: bool = False,
    limit: int | None = None,
) -> int:
    """
    Build or rebuild the persistent Chroma index
    from the parquet law dataset.
    """

    client = _client()

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = _collection()

    if collection.count() > 0 and not force:
        return collection.count()

    seen: set[tuple[str, str, str]] = set()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for batch_rows in _read_rows(limit=limit):
        for row_index, row in enumerate(batch_rows):
            entries = _iter_law_entries(row)

            for entry_index, entry in enumerate(entries):
                dedupe_key = (
                    entry.get("law_name", ""),
                    entry.get("section_num", ""),
                    entry.get("section_content", ""),
                )

                if dedupe_key in seen:
                    continue

                seen.add(dedupe_key)

                document_text = _build_document_text(entry)

                if not document_text:
                    continue

                doc_id = (
                    f"{row_index}_"
                    f"{entry_index}_"
                    f"{entry.get('section_num', 'x')}"
                )

                ids.append(doc_id)

                documents.append(
                    _passage_text(document_text)
                )

                metadatas.append(
                    {
                        "row_index": str(row_index),
                        "entry_index": str(entry_index),
                        "law_name": entry.get("law_name", ""),
                        "section_num": entry.get("section_num", ""),
                    }
                )

                if len(documents) >= BATCH_SIZE:
                    _add_batch(
                        collection,
                        ids,
                        documents,
                        metadatas,
                    )

                    ids.clear()
                    documents.clear()
                    metadatas.clear()

    _add_batch(
        collection,
        ids,
        documents,
        metadatas,
    )

    return collection.count()


def ensure_index(limit: int | None = None) -> None:
    collection = _collection()

    if collection.count() == 0:
        build_index(
            force=False,
            limit=limit,
        )


def search(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    ensure_index()

    collection = _collection()

    result = collection.query(
        query_texts=[_query_text(query)],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = result.get("documents", [[]])[0] or []
    metadatas = result.get("metadatas", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []

    matches: list[dict[str, Any]] = []

    for index, document in enumerate(documents):
        matches.append(
            {
                "document": document.replace(
                    "passage: ",
                    "",
                    1,
                ),
                "metadata": (
                    metadatas[index]
                    if index < len(metadatas)
                    else {}
                ),
                "distance": (
                    distances[index]
                    if index < len(distances)
                    else None
                ),
            }
        )

    return matches


def lookup_sections(
    section_numbers: list[int],
) -> list[str]:
    ensure_index()

    collection = _collection()

    fetched = collection.get(
        where={
            "section_num": {
                "$in": [str(n) for n in section_numbers]
            }
        },
        include=[
            "documents",
            "metadatas",
        ],
    )

    documents = fetched.get("documents", []) or []
    metadatas = fetched.get("metadatas", []) or []

    results: list[str] = []

    for index, document in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        law_name = metadata.get("law_name", "")
        section_num = metadata.get("section_num", "")

        cleaned_document = document.replace(
            "passage: ",
            "",
            1,
        )

        results.append(
            f"{law_name} "
            f"มาตรา {section_num}: "
            f"{cleaned_document}"
        )

    return results