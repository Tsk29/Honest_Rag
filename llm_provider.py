"""Single place that decides which LLM backs the app, so app.py and
workflow.py don't each hardcode a provider. Both already needed the same
construction logic - app.py for the UI-level cached LLM, workflow.py as its
fallback when no `llm` is passed in - so it lived in two places and would
have silently drifted the moment a second provider was added.

Set LLM_PROVIDER to switch:
  - "groq" (default) - fast, generous free tier, no local setup
  - "ollama" - fully local, e.g. Qwen2.5 via `ollama pull qwen2.5:7b`;
    trades speed/quality for zero API dependency and zero cost
  - "openai" - for parity with the original upstream project
"""
import os

from llama_index.core.llms import LLM


def build_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout=120.0,
        )

    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    from llama_index.llms.groq import Groq
    return Groq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=os.getenv("GROQ_API_KEY"),
    )
