"""Where the knowledge base lives, and what's in it.

The vector store is the single source of truth for which documents are
indexed. There used to be a separate JSON manifest next to the Milvus Lite
file; on a host with no persistent disk (e.g. a free Hugging Face Space) that
file is wiped on every restart while a hosted Milvus keeps the vectors, so the
two drift apart. Deriving the source list from the stored chunks' metadata
means there is nothing to keep in sync.

Connection is configured entirely through the environment, so the same code
runs against an embedded Milvus Lite file locally and a hosted Milvus
(Zilliz Cloud) in production:

  HONESTRAG_MILVUS_URI    - unset: <HONESTRAG_DATA_DIR>/milvus_demo.db (Milvus Lite)
                            or e.g. https://<cluster>.zillizcloud.com
  HONESTRAG_MILVUS_TOKEN  - API key for a hosted cluster; empty for Milvus Lite

Deliberately not plain MILVUS_URI: pymilvus reads that variable itself at
import time and rejects anything that isn't an http(s) address, so a local
file path there crashes the app before it starts.
"""

import os
from collections.abc import Iterable
from typing import Any

from pymilvus import MilvusClient

MILVUS_COLLECTION = "firecrawl_agent_docs"

# Stored on every chunk at index time. One Document per PDF page is what
# SimpleDirectoryReader produces, so the page count is known exactly then,
# but can't be recovered reliably afterwards (many PDFs carry duplicate or
# missing page labels).
PAGE_COUNT_KEY = "page_count"

_SOURCE_FIELDS = ["doc_id", "file_size", PAGE_COUNT_KEY, "page_label"]


def data_dir() -> str:
    return os.getenv("HONESTRAG_DATA_DIR", ".")


def milvus_uri() -> str:
    return os.getenv("HONESTRAG_MILVUS_URI") or os.path.join(data_dir(), "milvus_demo.db")


def milvus_token() -> str:
    return os.getenv("HONESTRAG_MILVUS_TOKEN", "")


def is_hosted() -> bool:
    return milvus_uri().startswith(("http://", "https://"))


def make_client() -> MilvusClient:
    return MilvusClient(uri=milvus_uri(), token=milvus_token())


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_sources(rows: Iterable[dict]) -> list[dict]:
    """Collapse per-chunk rows into one entry per source document:
    `{"name", "size_kb", "pages"}`, ordered by name.

    `pages` comes from the stored page_count; documents indexed before that
    field existed fall back to counting distinct page labels.
    """
    by_name: dict[str, dict] = {}
    for row in rows:
        name = row.get("doc_id")
        if not name:
            continue
        entry = by_name.setdefault(name, {"size": None, "page_count": None, "labels": set()})
        entry["size"] = entry["size"] or _to_int(row.get("file_size"))
        entry["page_count"] = entry["page_count"] or _to_int(row.get(PAGE_COUNT_KEY))
        if row.get("page_label") is not None:
            entry["labels"].add(row["page_label"])

    return [
        {
            "name": name,
            "size_kb": round(e["size"] / 1024) if e["size"] else None,
            "pages": e["page_count"] or (len(e["labels"]) or None),
        }
        for name, e in sorted(by_name.items())
    ]


def list_sources(client: MilvusClient, collection: str = MILVUS_COLLECTION) -> list[dict]:
    """Every document currently indexed in `collection`. Pages through the
    chunks with an iterator so it isn't capped by Milvus's per-query limit."""
    if not client.has_collection(collection):
        return []
    client.load_collection(collection)

    iterator = client.query_iterator(collection, batch_size=1000, filter="", output_fields=_SOURCE_FIELDS)
    rows: list[dict] = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    return summarize_sources(rows)


def clear(client: MilvusClient, collection: str = MILVUS_COLLECTION) -> None:
    """Drop every indexed document. Dropping the collection (rather than
    deleting the Milvus Lite file) works the same for a local file and a
    hosted cluster; the collection is recreated on the next upload."""
    if client.has_collection(collection):
        client.drop_collection(collection)
