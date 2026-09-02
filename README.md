# HonestRAG

A Corrective RAG (Retrieval-Augmented Generation) system that doesn't just trust
its own retrieval. It grades the relevance of what it pulled from your
documents, and automatically falls back to a live web search (via Firecrawl)
whenever that retrieval isn't good enough to answer confidently - then
generates a streamed, cited answer from whichever sources actually deserved
to be used.

*Originally forked from [patchy631/ai-engineering-hub's firecrawl-agent](https://github.com/patchy631/ai-engineering-hub/tree/main/firecrawl-agent),
then substantially rebuilt: concurrent relevance grading, a confidence-tiered
web-search trigger, source citations, real token streaming, a Groq-backed LLM
layer, multi-document persistent indexing with per-source removal, a
relevance-grader evaluation harness, and a NotebookLM-inspired UI.*

## Features

**Corrective RAG pipeline**
- Retrieves from your uploaded documents first, never the web by default
- Grades every retrieved chunk's relevance concurrently (all nodes graded in
  parallel via `asyncio.gather`, not one LLM call at a time)
- Confidence-tiered trigger: falls back to a live Firecrawl web search only
  when the *fraction* of relevant chunks drops below a threshold, instead of
  nuking a mostly-good retrieval because one chunk out of ten was noisy
- Blends document and web context when both are used, and tells you which

**Knowledge base**
- Upload multiple PDFs at once; each is embedded (FastEmbed) and stored in a
  persistent local Milvus index that survives app restarts
- Sources panel lists every indexed document with its size and page count
- Remove a single source without touching the rest of the knowledge base, or
  wipe everything via an explicit, checkbox-gated "Clear knowledge base"

**Answers you can verify**
- Real token-by-token streaming as the LLM generates (not a fake post-hoc
  reveal) via `llm.astream_complete()`
- Every answer is tagged with the sources it actually drew on - "Uploaded
  source" and/or the specific web URL(s) - shown as citation chips
- Runs on Groq by default (fast, generous free tier), swappable for OpenAI,
  Ollama, or LMStudio

**Interface**
- Streamlit UI restyled in a NotebookLM-inspired layout: dark theme, a
  dedicated Sources rail, a compact top bar, and an empty-state welcome card
  instead of a logo banner

**Measurement**
- A small evaluation harness (`eval/`) scores the relevance grader itself on
  a hand-labeled dataset - precision, recall, F1, accuracy - because the
  grader's yes/no call is the single most trust-critical judgment in the
  whole pipeline

## How it works

1. **Retrieve** - your query is embedded and matched against the persistent
   Milvus index of everything you've uploaded
2. **Grade** - every retrieved chunk is graded "relevant" or "not relevant" to
   your specific question, concurrently
3. **Decide** - if the relevant fraction clears the threshold, retrieval is
   trusted as-is; otherwise the query is rewritten and sent to Firecrawl for a
   live web search
4. **Generate** - the LLM answers from whichever context survived (document,
   web, or both), streaming tokens back as they're produced
5. **Cite** - the answer is returned with the specific sources it drew on, so
   you can check its work

![Workflow Architecture](assets/animation.gif)

## Tech Stack

| Layer | Choice |
|---|---|
| RAG framework | [LlamaIndex](https://github.com/run-llama/llama_index) (Workflow-based orchestration) |
| LLM | [Groq](https://groq.com/) by default (also OpenAI / Ollama / LMStudio) |
| Web search | [Firecrawl](https://firecrawl.dev/) |
| Vector store | [Milvus](https://milvus.io/) (Milvus Lite, local file-based) |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-large-en-v1.5`) |
| UI | [Streamlit](https://streamlit.io/) |

## Setup and Installation

### Prerequisites

- Python 3.11+
- A [Firecrawl](https://firecrawl.dev/) API key
- A [Groq](https://console.groq.com/) API key (or another provider - see
  `load_llm()` in `app.py`)

### 1. Install dependencies

An isolated virtual environment is recommended - this project's dependency
tree (LlamaIndex + several integration packages) is picky about version
pinning:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install llama-index-vector-stores-milvus llama-index-llms-groq
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
FIRECRAWL_API_KEY="your_firecrawl_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-120b"  # optional, defaults to this
```

Groq's available model lineup changes fairly often - if `GROQ_MODEL` starts
erroring out as deprecated, check what's currently live at
`https://api.groq.com/openai/v1/models`.

### 3. Run it

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (default `http://localhost:8501`), add a
PDF or two in the Sources panel, and start asking questions.

## Project Structure

```
Honestrag/
├── app.py                 # Streamlit UI: sources panel, chat, streaming
├── workflow.py             # CorrectiveRAGWorkflow: retrieve/grade/search/answer
├── eval/
│   ├── dataset.jsonl        # Labeled examples for the relevance grader
│   └── run_eval.py          # Precision/recall/F1 harness for the grader
├── start_server.py          # Optional Beam Cloud deployment (unmodified from
│                             #   upstream; not part of the local Groq setup above)
├── requirements.txt
└── assets/                  # Architecture diagram and logos used in this README
```

## Configuration

- **LLM**: set via `GROQ_API_KEY` / `GROQ_MODEL` in `.env`; swap providers by
  editing `load_llm()` in `app.py` and the matching default in `workflow.py`
- **Relevance threshold**: `CorrectiveRAGWorkflow.RELEVANCE_TRIGGER_THRESHOLD`
  in `workflow.py` (default `0.7`) - the minimum fraction of retrieved chunks
  that must grade "relevant" before local retrieval is trusted without a web
  search
- **Embedding model**: FastEmbed, `BAAI/bge-large-en-v1.5` by default
- **Vector store location**: `./milvus_demo.db` (gitignored - local only)

## Troubleshooting

1. **API key errors** - confirm `FIRECRAWL_API_KEY` and `GROQ_API_KEY` are set
   in `.env` and that the process picked it up (`load_dotenv()` runs at
   startup)
2. **`GROQ_MODEL` not found / deprecated** - check
   `https://api.groq.com/openai/v1/models` for the current live model list
3. **`ModuleNotFoundError: llama_index.vector_stores.milvus`** - this package
   isn't in `requirements.txt` (a gap inherited from the upstream repo);
   install it explicitly as shown in Setup step 1
4. **Vector store looking stale or corrupted** - delete `milvus_demo.db` and
   `milvus_demo_docs.json`, then re-upload your documents

## Evaluation

The Corrective RAG behavior in this app hinges on one judgment: for each
retrieved document chunk, the relevance grader (`DEFAULT_RELEVANCY_PROMPT_TEMPLATE`
in `workflow.py`) decides "yes" or "no" on whether that chunk is actually
relevant to the user's question. That decision determines whether the app
trusts local retrieval or falls back to a Firecrawl web search - it's the
single most trust-critical judgment in the pipeline.

`eval/` contains a small evaluation harness for that grader:

- `eval/dataset.jsonl` - ~20 hand-written labeled examples (`query`,
  `document_chunk`, `label`), including obviously relevant/irrelevant cases
  and several deliberately tricky near-misses (chunks that share keywords
  with the query but don't actually answer it, or are topically adjacent but
  not on-point). The easy cases don't tell you much about grader quality -
  the near-misses do.
- `eval/run_eval.py` - loads the dataset, imports the real
  `DEFAULT_RELEVANCY_PROMPT_TEMPLATE` from `workflow.py` (never reimplements
  it), runs it through an LLM, parses yes/no the same way the workflow does
  (including stripping `<think>` blocks), and reports precision, recall, F1,
  and accuracy - plus prints every misclassified example so you can see
  *which* cases the grader gets wrong, not just an aggregate score.

### Running it for real

```bash
OPENAI_API_KEY="your_openai_api_key_here" python eval/run_eval.py
```

Optional flags: `--model` (default `gpt-4o`) and `--dataset` to point at a
different labeled file. The eval harness still runs against OpenAI
independently of the app's own Groq-backed grader, so it can score the
grader's prompt/logic against a fixed, well-understood model.

### Mock mode

```bash
python eval/run_eval.py --mock
```

`--mock` (also triggered automatically if `OPENAI_API_KEY` isn't set) swaps
in a crude deterministic keyword-overlap heuristic instead of a real LLM
call. It exists solely so the harness's own plumbing - dataset loading,
prompt formatting, and metrics computation - can be verified without an API
key. Mock-mode numbers say nothing about the real grader's quality; only a
run with a real `OPENAI_API_KEY` is a meaningful evaluation.

## Acknowledgments

- [patchy631/ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub/tree/main/firecrawl-agent) for the original firecrawl-agent this was forked from
- [LlamaIndex](https://github.com/run-llama/llama_index) for the RAG framework
- [Firecrawl](https://firecrawl.dev/) for web search
- [Groq](https://groq.com/) for LLM inference
- [Milvus](https://milvus.io/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface
