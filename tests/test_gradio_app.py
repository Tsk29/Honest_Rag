import gradio as gr
import pytest
from llama_index.core.embeddings import MockEmbedding

import gradio_app
import rag_service
from tests.conftest import ANSWER_TEXT, FakeIndex, ScriptedLLM
from workflow import CorrectiveRAGWorkflow

# --- presentation helpers ----------------------------------------------------


def test_sources_table_blanks_unknown_values():
    docs = [{"name": "a.pdf", "size_kb": 12, "pages": 3}, {"name": "b.pdf", "size_kb": None, "pages": None}]
    assert gradio_app.sources_table(docs) == [["a.pdf", 12, 3], ["b.pdf", "", ""]]


def test_citations_escape_html_and_number_multiple_documents():
    sources = [
        {"type": "document", "text": "<script>alert(1)</script>"},
        {"type": "document", "text": "second"},
        {"type": "web", "url": "https://example.com", "text": "web text"},
    ]
    out = gradio_app.format_citations(sources)

    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "Uploaded source 1" in out and "Uploaded source 2" in out
    assert "🌐 https://example.com" in out


def test_citations_truncate_long_snippets():
    out = gradio_app.format_citations([{"type": "document", "text": "x" * 5000}])
    assert "x" * gradio_app.MAX_SNIPPET_CHARS + "…" in out
    assert "x" * (gradio_app.MAX_SNIPPET_CHARS + 1) not in out


def test_critique_formats():
    assert gradio_app.format_critique(None) == ""
    assert "Claims checked" in gradio_app.format_critique({"passed": True, "note": None})
    failed = gradio_app.format_critique({"passed": False, "note": "- <b>made up</b>"})
    assert "may not be fully supported" in failed and "&lt;b&gt;made up&lt;/b&gt;" in failed


def test_final_answer_without_extras_is_just_the_answer():
    assert gradio_app.format_final_answer({"answer": "Hi", "sources": [], "critique": None}) == "Hi"


def test_zerogpu_placeholder_registered_when_spaces_runtime_present(monkeypatch):
    import importlib
    import sys
    import types

    registered = []
    fake = types.ModuleType("spaces")
    fake.GPU = lambda **kw: (lambda fn: registered.append((fn.__name__, kw)) or fn)
    monkeypatch.setitem(sys.modules, "spaces", fake)
    try:
        importlib.reload(gradio_app)
        assert registered == [("_zerogpu_placeholder", {"duration": 1})]
    finally:
        monkeypatch.delitem(sys.modules, "spaces")
        importlib.reload(gradio_app)


def test_ui_builds():
    assert isinstance(gradio_app.build_ui(), gr.Blocks)


# --- chat handler ------------------------------------------------------------


async def _collect(gen):
    return [chunk async for chunk in gen]


async def test_respond_without_sources_asks_for_an_upload(monkeypatch):
    monkeypatch.setattr(gradio_app.KB, "ensure_workflow", lambda: None)
    assert await _collect(gradio_app.respond("hi", [])) == [gradio_app.NO_SOURCES_MESSAGE]


async def test_respond_streams_then_shows_answer_with_verdict_and_citations(monkeypatch):
    wf = CorrectiveRAGWorkflow(index=FakeIndex(["RELEVANT fact"]), firecrawl_api_key="k", llm=ScriptedLLM(prompts=[]))
    monkeypatch.setattr(gradio_app.KB, "ensure_workflow", lambda: wf)

    chunks = await _collect(gradio_app.respond("What is the capital?", []))

    assert len(chunks) > 2 and chunks[0].endswith("▌")  # streamed progressively
    final = chunks[-1]
    assert final.startswith(ANSWER_TEXT)
    assert "Claims checked" in final and "Uploaded source" in final


async def test_respond_reports_workflow_errors(monkeypatch):
    class Broken:
        def run(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(gradio_app.KB, "ensure_workflow", lambda: Broken())
    chunks = await _collect(gradio_app.respond("q", []))
    assert chunks == ["⚠️ An error occurred while answering: boom"]


# --- knowledge base end to end (Milvus Lite + mock embeddings) ----------------


@pytest.fixture
def kb_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HONESTRAG_MILVUS_URI", str(tmp_path / "kb.db"))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test")
    monkeypatch.setattr(rag_service, "build_embed_model", lambda: MockEmbedding(embed_dim=rag_service.EMBED_DIM))
    monkeypatch.setattr(gradio_app, "build_llm", lambda: ScriptedLLM(prompts=[]))
    kb_obj = gradio_app.KnowledgeBase()
    monkeypatch.setattr(gradio_app, "KB", kb_obj)
    yield kb_obj, tmp_path
    kb_obj.client.close()


def _upload(tmp_path, name, text):
    # Gradio stores each upload in its own temp dir under the original name.
    d = tmp_path / f"gradio-{name}"
    d.mkdir()
    (d / name).write_text(text)
    return str(d / name)


def test_upload_remove_clear_through_handlers(kb_env):
    kb_obj, tmp = kb_env
    a = _upload(tmp, "a.txt", "alpha " * 50)
    b = _upload(tmp, "b.txt", "beta " * 50)

    rows, dropdown, status, cleared_upload = gradio_app.on_upload([a, b])
    assert [r[0] for r in rows] == ["a.txt", "b.txt"]
    assert "Added: a.txt, b.txt" in status and cleared_upload is None
    assert kb_obj.workflow is not None

    _, _, status, _ = gradio_app.on_upload([a])
    assert "already indexed" in status

    rows, _, status = gradio_app.on_remove("a.txt")
    assert [r[0] for r in rows] == ["b.txt"] and "Removed: a.txt" in status

    rows, _, status, confirm = gradio_app.on_clear(False)
    assert rows and "confirmation" in status  # nothing deleted without the checkbox

    rows, _, status, confirm = gradio_app.on_clear(True)
    assert rows == [] and confirm is False and kb_obj.workflow is None


def test_workflow_resumes_from_existing_index_after_restart(kb_env):
    kb_obj, tmp = kb_env
    kb_obj.add_files([_upload(tmp, "a.txt", "alpha " * 50)])

    fresh = gradio_app.KnowledgeBase()  # e.g. the Space restarted
    try:
        assert fresh.ensure_workflow() is not None
    finally:
        fresh.client.close()
