import pytest
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llm_provider import build_llm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "GROQ_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")


def test_defaults_to_groq():
    llm = build_llm()
    assert isinstance(llm, Groq)
    assert llm.model == "openai/gpt-oss-120b"


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
    assert isinstance(build_llm(), Ollama)


def test_ollama_honours_model_and_base_url(monkeypatch):
    # The Docker Compose setup relies on OLLAMA_BASE_URL pointing at the
    # `ollama` service instead of localhost.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")

    llm = build_llm()

    assert llm.model == "qwen2.5:7b"
    assert llm.base_url == "http://ollama:11434"


def test_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    llm = build_llm()

    assert isinstance(llm, OpenAI)
    assert llm.model == "gpt-4o-mini"
