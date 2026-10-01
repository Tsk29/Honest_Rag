import pytest
from llama_index.core.embeddings import MockEmbedding

import knowledge_base as kb
import rag_service


@pytest.fixture
def milvus_lite(tmp_path, monkeypatch):
    """Point the knowledge base at a throwaway embedded Milvus Lite file."""
    monkeypatch.setenv("HONESTRAG_MILVUS_URI", str(tmp_path / "kb.db"))
    monkeypatch.delenv("HONESTRAG_MILVUS_TOKEN", raising=False)
    client = kb.make_client()
    yield client
    client.close()


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "upload"
    d.mkdir()
    (d / "paris.txt").write_text("Paris is the capital of France. " * 20)
    (d / "rome.txt").write_text("Rome is the capital of Italy. " * 20)
    return d


def test_load_documents_tags_each_doc_with_filename_and_page_count(docs_dir):
    docs = rag_service.load_documents(str(docs_dir))

    assert sorted(d.doc_id for d in docs) == ["paris.txt", "rome.txt"]
    for d in docs:
        assert d.metadata[kb.PAGE_COUNT_KEY] == 1
        # bookkeeping must not leak into what's embedded or shown to the LLM
        assert kb.PAGE_COUNT_KEY in d.excluded_embed_metadata_keys
        assert kb.PAGE_COUNT_KEY in d.excluded_llm_metadata_keys


def test_index_list_and_remove_round_trip(docs_dir, milvus_lite):
    embed = MockEmbedding(embed_dim=rag_service.EMBED_DIM)
    store = rag_service.open_vector_store()

    rag_service.index_documents(rag_service.load_documents(str(docs_dir)), store, embed)
    assert [s["name"] for s in kb.list_sources(milvus_lite)] == ["paris.txt", "rome.txt"]

    rag_service.remove_source(store, "paris.txt")
    assert [s["name"] for s in kb.list_sources(milvus_lite)] == ["rome.txt"]


def test_reopening_the_store_keeps_existing_documents(docs_dir, milvus_lite):
    embed = MockEmbedding(embed_dim=rag_service.EMBED_DIM)
    rag_service.index_documents(rag_service.load_documents(str(docs_dir)), rag_service.open_vector_store(), embed)

    rag_service.open_vector_store()  # overwrite=False must not drop the collection

    assert len(kb.list_sources(milvus_lite)) == 2


def test_build_workflow_requires_firecrawl_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
        rag_service.build_workflow(vector_store=None, embed_model=None, llm=None)
