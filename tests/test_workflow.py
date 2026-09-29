import requests
from llama_index.core.schema import NodeWithScore, TextNode

from tests.conftest import ANSWER_TEXT, FakeIndex
from workflow import CorrectiveRAGWorkflow, TokenEvent, _tag_web_search_results

WEB_RESULTS = (
    "Title: France\nDescription: Paris is the capital.\nURL: https://example.com/france\n"
    "\n---\n"
    "Title: Europe\nDescription: Capitals of Europe.\nURL: https://example.com/europe\n"
)


def _make_workflow(texts, llm, monkeypatch, web_results=WEB_RESULTS):
    wf = CorrectiveRAGWorkflow(index=FakeIndex(texts), firecrawl_api_key="test-key", llm=llm, timeout=30)
    searches: list[str] = []

    def fake_search(query, limit=5):
        searches.append(query)
        return web_results

    monkeypatch.setattr(wf, "_firecrawl_search", fake_search)
    return wf, searches


# --- _tag_web_search_results -------------------------------------------------

def test_tag_web_search_results_tags_each_block_with_its_url():
    tagged = _tag_web_search_results(WEB_RESULTS)
    assert "[Web Source: https://example.com/france]\nTitle: France" in tagged
    assert "[Web Source: https://example.com/europe]\nTitle: Europe" in tagged


def test_tag_web_search_results_handles_missing_url_and_empty_input():
    assert _tag_web_search_results("   ") == "   "
    assert _tag_web_search_results("Title: no url here").startswith("[Web Source: unknown]")


# --- relevance threshold routing ---------------------------------------------

async def test_mostly_relevant_retrieval_skips_web_search(scripted_llm, monkeypatch):
    # 3/4 relevant = 0.75 >= 0.7 threshold -> trust local retrieval
    texts = ["RELEVANT a", "RELEVANT b", "RELEVANT c", "IRRELEVANT d"]
    wf, searches = _make_workflow(texts, scripted_llm, monkeypatch)

    result = await wf.run(query_str="What is the capital of France?")

    assert searches == []
    assert result["answer"] == ANSWER_TEXT
    assert [s["type"] for s in result["sources"]] == ["document"] * 3
    assert result["critique"] == {"passed": True, "note": None}


async def test_mostly_irrelevant_retrieval_falls_back_to_web_search(scripted_llm, monkeypatch):
    # 1/4 relevant = 0.25 < 0.7 threshold -> rewrite query and search the web
    texts = ["RELEVANT a", "IRRELEVANT b", "IRRELEVANT c", "IRRELEVANT d"]
    wf, searches = _make_workflow(texts, scripted_llm, monkeypatch)

    result = await wf.run(query_str="What is the capital of France?")

    assert searches == ["rewritten query"]
    types = [s["type"] for s in result["sources"]]
    assert types == ["document", "web", "web"]
    assert {s["url"] for s in result["sources"] if s["type"] == "web"} == {
        "https://example.com/france",
        "https://example.com/europe",
    }


async def test_no_context_at_all_returns_canned_answer(scripted_llm, monkeypatch):
    wf, searches = _make_workflow(["IRRELEVANT"], scripted_llm, monkeypatch, web_results="")

    result = await wf.run(query_str="anything")

    assert searches == ["rewritten query"]
    assert result == {
        "answer": "No relevant information found in the documents.",
        "sources": [],
        "critique": None,
    }


# --- streaming + critique ----------------------------------------------------

async def test_answer_is_streamed_as_token_events(scripted_llm, monkeypatch):
    wf, _ = _make_workflow(["RELEVANT a"], scripted_llm, monkeypatch)

    handler = wf.run(query_str="q")
    deltas = [ev.delta async for ev in handler.stream_events() if isinstance(ev, TokenEvent)]
    result = await handler

    assert len(deltas) > 1
    assert "".join(deltas) == result["answer"] == ANSWER_TEXT


async def test_failed_critique_surfaces_the_unsupported_claims(scripted_llm, monkeypatch):
    scripted_llm.critique_reply = "- Claims Paris has 40 million people"
    wf, _ = _make_workflow(["RELEVANT a"], scripted_llm, monkeypatch)

    result = await wf.run(query_str="q")

    assert result["critique"] == {"passed": False, "note": "- Claims Paris has 40 million people"}


async def test_grader_failure_is_treated_as_not_relevant(scripted_llm, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    wf, _ = _make_workflow(["RELEVANT a"], scripted_llm, monkeypatch)
    monkeypatch.setattr(type(scripted_llm), "complete", boom)

    node = NodeWithScore(node=TextNode(text="RELEVANT"), score=1.0)
    assert await wf._grade_node_relevance(0, node, "q") == "no"


# --- Firecrawl client --------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_firecrawl_search_formats_results(scripted_llm, monkeypatch):
    payload = {"success": True, "data": [{"title": "T", "description": "D", "url": "https://u"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(payload))
    wf = CorrectiveRAGWorkflow(index=FakeIndex([]), firecrawl_api_key="k", llm=scripted_llm)

    assert wf._firecrawl_search("q") == "Title: T\nDescription: D\nURL: https://u\n"


def test_firecrawl_search_swallows_network_errors(scripted_llm, monkeypatch):
    def fail(*a, **k):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(requests, "post", fail)
    wf = CorrectiveRAGWorkflow(index=FakeIndex([]), firecrawl_api_key="k", llm=scripted_llm)

    assert wf._firecrawl_search("q") == ""
