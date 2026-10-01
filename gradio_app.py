"""Gradio front end for HonestRAG, used for the Hugging Face Space.

Same pipeline as the Streamlit app (app.py): indexing, retrieval, grading,
web fallback and self-critique all come from rag_service.py / workflow.py.
Only the UI differs, because free Hugging Face Spaces host Gradio apps but not
Docker/Streamlit ones.

State is process-wide, not per browser session: every visitor shares one
knowledge base (the same is true of the Streamlit app). Deploy the Space as
private.

Run locally:  uv run --extra space python gradio_app.py
"""

import html
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator

import gradio as gr
from dotenv import load_dotenv
from llama_index.core import Settings

import knowledge_base as kb
import rag_service
from llm_provider import build_llm
from workflow import TokenEvent

load_dotenv()
logger = logging.getLogger("honestrag")

# Free Hugging Face Gradio Spaces run on ZeroGPU hardware, whose `spaces`
# runtime (pre-installed there, absent locally) expects the app to register at
# least one @spaces.GPU function at startup. HonestRAG needs no GPU - embeddings
# run on CPU via ONNX and the LLM is a hosted API - so this placeholder is never
# called and never uses any of the account's GPU quota.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder() -> None:
        """Registered for ZeroGPU's startup check only; never called."""

MAX_SNIPPET_CHARS = 1500
NO_SOURCES_MESSAGE = "Add a PDF in the **Sources** panel first, then ask a question about it."


# --- presentation helpers (pure, unit tested) --------------------------------

def sources_table(docs: list[dict]) -> list[list]:
    """Rows for the Sources dataframe."""
    return [[d["name"], d.get("size_kb") or "", d.get("pages") or ""] for d in docs]


def format_citations(sources: list[dict]) -> str:
    """Each source the answer could draw on, as a collapsed block holding the
    exact retrieved snippet - so a claim can be checked, not taken on faith."""
    if not sources:
        return ""
    n_docs = sum(1 for s in sources if s.get("type") == "document")
    blocks, doc_i = [], 0
    for source in sources:
        if source.get("type") == "document":
            doc_i += 1
            label = f"📄 Uploaded source {doc_i}" if n_docs > 1 else "📄 Uploaded source"
        else:
            label = f"🌐 {source.get('url', 'web')}"
        text = source.get("text", "")
        if len(text) > MAX_SNIPPET_CHARS:
            text = text[:MAX_SNIPPET_CHARS] + "…"
        blocks.append(
            f"<details><summary>{html.escape(label)}</summary>\n\n"
            f"<blockquote>{html.escape(text)}</blockquote>\n\n</details>"
        )
    return "\n".join(blocks)


def format_critique(critique: dict | None) -> str:
    """The self-critique verdict. Empty when the check couldn't run at all,
    rather than implying a result that was never computed."""
    if critique is None:
        return ""
    if critique.get("passed"):
        return "✓ *Claims checked against sources*"
    note = html.escape(critique.get("note") or "")
    return (
        "<details><summary>⚠️ Some claims may not be fully supported by sources</summary>\n\n"
        f"<blockquote>{note}</blockquote>\n\n</details>"
    )


def format_final_answer(result: dict) -> str:
    parts = [result.get("answer", "")]
    critique = format_critique(result.get("critique"))
    citations = format_citations(result.get("sources", []))
    if critique or citations:
        parts.append("---")
    parts.extend(p for p in (critique, citations) if p)
    return "\n\n".join(parts)


# --- knowledge base state ----------------------------------------------------

class KnowledgeBase:
    """Process-wide handles: embedding model, LLM, vector store, workflow.
    Everything heavy is created lazily on first use so the UI comes up fast,
    and mutations are serialised with a lock (Gradio runs handlers in threads).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embed_model = None
        self._llm = None
        self._client = None
        self.workflow = None

    def _ensure_models(self):
        if self._embed_model is None:
            self._embed_model = rag_service.build_embed_model()
            Settings.embed_model = self._embed_model
        if self._llm is None:
            self._llm = build_llm()
            Settings.llm = self._llm

    @property
    def client(self):
        if self._client is None:
            self._client = kb.make_client()
        return self._client

    def sources(self) -> list[dict]:
        return kb.list_sources(self.client)

    def _rebuild_workflow(self, vector_store) -> None:
        self.workflow = (
            rag_service.build_workflow(vector_store, self._embed_model, self._llm) if self.sources() else None
        )

    def ensure_workflow(self):
        """Resume from whatever is already indexed (e.g. after a restart)."""
        with self._lock:
            if self.workflow is None and self.sources():
                self._ensure_models()
                self._rebuild_workflow(rag_service.open_vector_store())
            return self.workflow

    def add_files(self, paths: list[str]) -> list[str]:
        """Index files not already in the knowledge base; returns names added."""
        existing = {d["name"] for d in self.sources()}
        new_paths = [p for p in paths if os.path.basename(p) not in existing]
        if not new_paths:
            return []
        with self._lock, tempfile.TemporaryDirectory() as tmp:
            # Gradio uploads land in per-file temp dirs; gather this batch into
            # one directory so SimpleDirectoryReader reads exactly these files.
            for p in new_paths:
                shutil.copy(p, os.path.join(tmp, os.path.basename(p)))
            documents = rag_service.load_documents(tmp)
            self._ensure_models()
            vector_store = rag_service.open_vector_store()
            rag_service.index_documents(documents, vector_store, self._embed_model)
            self._rebuild_workflow(vector_store)
        return [os.path.basename(p) for p in new_paths]

    def remove(self, name: str) -> None:
        with self._lock:
            self._ensure_models()
            vector_store = rag_service.open_vector_store()
            rag_service.remove_source(vector_store, name)
            self._rebuild_workflow(vector_store)

    def clear(self) -> None:
        with self._lock:
            kb.clear(self.client)
            self.workflow = None


KB = KnowledgeBase()


# --- event handlers ----------------------------------------------------------

def _sources_outputs(status: str):
    docs = KB.sources()
    names = [d["name"] for d in docs]
    return (
        sources_table(docs),
        gr.update(choices=names, value=None),
        f"{status}\n\n{len(docs)} source(s) indexed." if status else f"{len(docs)} source(s) indexed.",
    )


def on_load():
    try:
        return _sources_outputs("")
    except Exception as e:  # surface any connection error in the UI
        logger.exception("Could not read the knowledge base")
        return [], gr.update(choices=[]), f"⚠️ Could not connect to the knowledge base: {e}"


def on_upload(files: list[str] | None):
    if not files:
        return (*_sources_outputs(""), None)
    try:
        added = KB.add_files(files)
        status = f"Added: {', '.join(added)}" if added else "Those files are already indexed."
    except Exception as e:
        logger.exception("Indexing failed")
        status = f"⚠️ Failed to index: {e}"
    return (*_sources_outputs(status), None)


def on_remove(name: str | None):
    if not name:
        return _sources_outputs("Pick a source to remove.")
    KB.remove(name)
    return _sources_outputs(f"Removed: {name}")


def on_clear(confirmed: bool):
    if not confirmed:
        return (*_sources_outputs("Tick the confirmation box first."), False)
    KB.clear()
    return (*_sources_outputs("Knowledge base cleared."), False)


async def respond(message: str, history: list) -> AsyncIterator[str]:
    try:
        workflow = KB.ensure_workflow()
    except Exception as e:
        yield f"⚠️ Could not load the knowledge base: {e}"
        return
    if workflow is None:
        yield NO_SOURCES_MESSAGE
        return

    streamed = ""
    try:
        handler = workflow.run(query_str=message)
        async for event in handler.stream_events():
            if isinstance(event, TokenEvent):
                streamed += event.delta
                yield streamed + " ▌"
        result = await handler
    except Exception as e:
        logger.exception("Workflow failed")
        yield f"⚠️ An error occurred while answering: {e}"
        return

    yield format_final_answer(result) if isinstance(result, dict) else str(result)


# --- layout ------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HonestRAG", fill_height=True) as demo:
        gr.Markdown(
            "## 📓 HonestRAG\n"
            "Answers from your uploaded sources first, grades how relevant what it found actually is, "
            "falls back to a live web search when your sources don't cover the question, and fact-checks "
            "its own answer."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Sources")
                upload = gr.File(label="Add PDFs", file_count="multiple", file_types=[".pdf"], type="filepath")
                status = gr.Markdown()
                table = gr.Dataframe(headers=["Source", "KB", "Pages"], interactive=False, wrap=True)
                with gr.Row():
                    to_remove = gr.Dropdown(label="Remove a source", choices=[], scale=3)
                    remove_btn = gr.Button("Remove", scale=1)
                with gr.Accordion("Danger zone", open=False):
                    confirm = gr.Checkbox(label="I understand this permanently deletes all indexed documents")
                    clear_btn = gr.Button("Clear knowledge base", variant="stop")
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=respond,
                    chatbot=gr.Chatbot(height=600, placeholder="Ask a question about your documents."),
                    fill_height=True,
                )

        sources_outputs = [table, to_remove, status]
        demo.load(on_load, outputs=sources_outputs)
        upload.upload(on_upload, inputs=upload, outputs=[*sources_outputs, upload])
        remove_btn.click(on_remove, inputs=to_remove, outputs=sources_outputs)
        clear_btn.click(on_clear, inputs=confirm, outputs=[*sources_outputs, confirm])
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_ui().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"), theme=gr.themes.Soft())
