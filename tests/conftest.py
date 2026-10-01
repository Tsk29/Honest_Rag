"""Shared test doubles. Nothing in the test suite talks to a real LLM,
Firecrawl, or Milvus - CI runs without any API keys."""

from typing import Any

import pytest
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.schema import NodeWithScore, TextNode

ANSWER_TEXT = "Paris is the capital of France.\nSources: Document"


class ScriptedLLM(CustomLLM):
    """Deterministic stand-in for a real LLM. Routes on which of the
    workflow's prompts it was given, so a whole CorrectiveRAGWorkflow run
    can be driven end to end:

    - relevance grader: "yes" iff the chunk contains the marker RELEVANT
    - query rewrite:    a fixed rewritten query
    - self-critique:    `critique_reply` (SUPPORTED by default)
    - anything else:    the final answer, streamed a word at a time
    """

    critique_reply: str = "SUPPORTED"
    prompts: list[str] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="scripted-test-llm")

    def _reply(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "As a grader" in prompt:
            chunk = prompt.split("Retrieved Document:")[1].split("User Question:")[0]
            return "yes" if "RELEVANT" in chunk and "IRRELEVANT" not in chunk else "no"
        if "refine a query" in prompt:
            return "rewritten query"
        if "fact-checking" in prompt:
            return self.critique_reply
        return ANSWER_TEXT

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._reply(prompt))

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        text = self._reply(prompt)

        def gen() -> CompletionResponseGen:
            so_far = ""
            for word in text.split(" "):
                delta = word if not so_far else " " + word
                so_far += delta
                yield CompletionResponse(text=so_far, delta=delta)

        return gen()


class FakeRetriever:
    def __init__(self, texts: list[str]):
        self._texts = texts

    def retrieve(self, query_str: str) -> list[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text=t), score=1.0) for t in self._texts]


class FakeIndex:
    """Just enough of VectorStoreIndex for CorrectiveRAGWorkflow.retrieve()."""

    def __init__(self, texts: list[str]):
        self._texts = texts

    def as_retriever(self, **kwargs: Any) -> FakeRetriever:
        return FakeRetriever(self._texts)


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM(prompts=[])
