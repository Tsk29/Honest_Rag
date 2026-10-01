import os
import subprocess
import sys

import pytest
from pymilvus import DataType, MilvusClient

import knowledge_base as kb

# --- configuration -----------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("HONESTRAG_MILVUS_URI", "HONESTRAG_MILVUS_TOKEN", "HONESTRAG_DATA_DIR"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_defaults_to_local_milvus_lite_file(clean_env):
    assert kb.milvus_uri() == "./milvus_demo.db"
    assert kb.milvus_token() == ""
    assert not kb.is_hosted()


def test_local_file_follows_data_dir(clean_env):
    clean_env.setenv("HONESTRAG_DATA_DIR", "/data")
    assert kb.milvus_uri() == "/data/milvus_demo.db"


def test_hosted_milvus_from_env(clean_env):
    clean_env.setenv("HONESTRAG_MILVUS_URI", "https://in03-abc.serverless.gcp-us-west1.cloud.zilliz.com")
    clean_env.setenv("HONESTRAG_MILVUS_TOKEN", "secret")
    assert kb.is_hosted()
    assert kb.milvus_token() == "secret"


def test_local_file_uri_does_not_break_pymilvus_import(tmp_path):
    # Regression: under the name MILVUS_URI, pymilvus parsed a file path at
    # import time and crashed. Needs a fresh interpreter to re-run the import.
    env = {**os.environ, "HONESTRAG_MILVUS_URI": str(tmp_path / "kb.db")}
    code = "import knowledge_base as kb; c = kb.make_client(); print(c.list_collections()); c.close()"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]


# --- summarize_sources (pure) ------------------------------------------------


def test_summarize_groups_chunks_by_document():
    rows = [
        {"doc_id": "b.pdf", "file_size": 2048, "page_count": 3, "page_label": "1"},
        {"doc_id": "a.pdf", "file_size": "10240", "page_count": "5", "page_label": "1"},
        {"doc_id": "b.pdf", "file_size": 2048, "page_count": 3, "page_label": "2"},
    ]
    assert kb.summarize_sources(rows) == [
        {"name": "a.pdf", "size_kb": 10, "pages": 5},
        {"name": "b.pdf", "size_kb": 2, "pages": 3},
    ]


def test_summarize_falls_back_to_page_labels_for_legacy_chunks():
    # Indexed before page_count was stored: count distinct labels instead.
    rows = [{"doc_id": "old.pdf", "page_label": str(p)} for p in (1, 2, 2, 3)]
    assert kb.summarize_sources(rows) == [{"name": "old.pdf", "size_kb": None, "pages": 3}]


def test_summarize_ignores_rows_without_doc_id_and_handles_empty():
    assert kb.summarize_sources([{"text": "orphan"}]) == []
    assert kb.summarize_sources([]) == []


# --- against a real (embedded) Milvus Lite -----------------------------------

DIM = 4


@pytest.fixture
def client(tmp_path):
    c = MilvusClient(str(tmp_path / "test.db"))
    yield c
    c.close()


def _create_collection_like_llamaindex(client: MilvusClient, name: str) -> None:
    """Same shape MilvusVectorStore creates: fixed id/doc_id/text/embedding
    fields, everything else (file_size, page_count, ...) as dynamic fields."""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
    schema.add_field("text", DataType.VARCHAR, max_length=1024)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    index = client.prepare_index_params()
    index.add_index("embedding", metric_type="IP", index_type="FLAT")
    client.create_collection(name, schema=schema, index_params=index)


def test_list_sources_on_missing_collection_is_empty(client):
    assert kb.list_sources(client, "nope") == []


def test_list_sources_reads_back_what_was_indexed(client):
    _create_collection_like_llamaindex(client, "docs")
    rows = [
        {"id": f"a{i}", "doc_id": "a.pdf", "text": "x", "embedding": [0.1] * DIM,
         "file_size": 4096, "page_count": 2, "page_label": str(i)}
        for i in range(3)
    ] + [
        {"id": "b0", "doc_id": "b.pdf", "text": "y", "embedding": [0.2] * DIM,
         "file_size": 1024, "page_count": 1, "page_label": "1"}
    ]
    client.insert("docs", rows)

    assert kb.list_sources(client, "docs") == [
        {"name": "a.pdf", "size_kb": 4, "pages": 2},
        {"name": "b.pdf", "size_kb": 1, "pages": 1},
    ]


def test_clear_drops_everything_and_is_idempotent(client):
    _create_collection_like_llamaindex(client, "docs")
    client.insert("docs", [{"id": "a", "doc_id": "a.pdf", "text": "x", "embedding": [0.1] * DIM}])

    kb.clear(client, "docs")
    kb.clear(client, "docs")

    assert not client.has_collection("docs")
    assert kb.list_sources(client, "docs") == []
