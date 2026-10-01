"""UI-independent RAG plumbing shared by both front ends: the Streamlit app
(app.py - local / Docker) and the Gradio app (gradio_app.py - Hugging Face
Space). Loading and tagging documents, embedding them into Milvus, removing
them, and wiring a CorrectiveRAGWorkflow to the result all live here, so the
two UIs can't drift apart in how they index or answer.
"""

import os

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.schema import Document
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

import knowledge_base as kb
from workflow import CorrectiveRAGWorkflow

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024  # must match EMBED_MODEL_NAME; fixed when the collection is created

# Where the FastEmbed model is cached. The Docker image points this at a copy
# baked in at build time and sets HONESTRAG_EMBED_OFFLINE so it never checks
# Hugging Face for updates; elsewhere it's downloaded on first use.
EMBED_CACHE_DIR = os.getenv("HONESTRAG_EMBED_CACHE_DIR") or os.path.join(kb.data_dir(), "hf_cache")
EMBED_OFFLINE = os.getenv("HONESTRAG_EMBED_OFFLINE", "").lower() in ("1", "true", "yes")


def build_embed_model() -> FastEmbedEmbedding:
    return FastEmbedEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_dir=EMBED_CACHE_DIR,
        local_files_only=EMBED_OFFLINE,
    )


def open_vector_store() -> MilvusVectorStore:
    """Connect to the persistent collection, creating it on first use.

    `overwrite=False`: MilvusVectorStore only drops the collection when
    overwrite=True AND it already exists, so this gives "create on first use,
    reuse after that" and every previously uploaded document survives.
    """
    vector_store = MilvusVectorStore(
        uri=kb.milvus_uri(),
        token=kb.milvus_token(),
        collection_name=kb.MILVUS_COLLECTION,
        dim=EMBED_DIM,
        overwrite=False,
    )
    # MilvusVectorStore only calls load_collection() itself when it creates a
    # brand-new collection. A fresh connection to an *existing* one starts it
    # "released", and any search fails with "call load() before search".
    if kb.MILVUS_COLLECTION in vector_store.client.list_collections():
        vector_store.client.load_collection(kb.MILVUS_COLLECTION)
    return vector_store


def load_documents(directory: str) -> list[Document]:
    """Read every file in `directory` and tag each Document for the knowledge base.

    - doc_id := original filename. SimpleDirectoryReader derives it from the
      (temporary) upload path, which is useless for removing a source later;
      the filename gives every chunk a stable ref_doc_id to delete by.
    - page_count := number of Documents from that file. A PDF loads as one
      Document per page, so this is exact here and can't be recovered
      reliably afterwards (see knowledge_base.summarize_sources). It's
      bookkeeping only, so it's kept out of the embedded and LLM text.
    """
    documents = SimpleDirectoryReader(directory).load_data()

    page_counts: dict[str, int] = {}
    for doc in documents:
        name = doc.metadata.get("file_name", "")
        if name:
            page_counts[name] = page_counts.get(name, 0) + 1

    for doc in documents:
        name = doc.metadata.get("file_name", "")
        if not name:
            continue
        doc.doc_id = name
        doc.metadata[kb.PAGE_COUNT_KEY] = page_counts[name]
        doc.excluded_embed_metadata_keys.append(kb.PAGE_COUNT_KEY)
        doc.excluded_llm_metadata_keys.append(kb.PAGE_COUNT_KEY)
    return documents


def index_documents(documents: list[Document], vector_store: MilvusVectorStore, embed_model) -> None:
    """Embed `documents` and insert them into the existing collection."""
    VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=embed_model,
    )


def remove_source(vector_store: MilvusVectorStore, name: str) -> None:
    """Delete one source's chunks without touching any other document."""
    vector_store.delete(ref_doc_id=name)


def build_workflow(vector_store: MilvusVectorStore, embed_model, llm: LLM) -> CorrectiveRAGWorkflow:
    """A workflow that retrieves across *everything* in the collection.

    Built with `from_vector_store`, not the index returned by
    `from_documents`: the latter restricts retrieval to the nodes inserted in
    that one call, which would silently limit answers to the latest upload.
    """
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        raise ValueError("FIRECRAWL_API_KEY is not set.")

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    return CorrectiveRAGWorkflow(
        index=index,
        firecrawl_api_key=firecrawl_api_key,
        llm=llm,
        verbose=True,
        timeout=249,
    )
