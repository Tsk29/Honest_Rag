# HonestRAG

A Corrective RAG (Retrieval-Augmented Generation) system that doesn't just trust
its own retrieval - or its own answer. It grades the relevance of what it
pulled from your documents, falls back to a live web search (via Firecrawl)
whenever that retrieval isn't good enough to answer confidently, then
fact-checks its own finished answer against the context it used before
showing it to you, with the exact source snippet behind every claim one
click away.

*Originally forked from [patchy631/ai-engineering-hub's firecrawl-agent](https://github.com/patchy631/ai-engineering-hub/tree/main/firecrawl-agent),
then substantially rebuilt: concurrent relevance grading, a confidence-tiered
web-search trigger, a self-critique pass on every answer, per-snippet source
citations, real token streaming, a pluggable LLM layer (Groq / local Ollama /
OpenAI), multi-document persistent indexing with per-source removal, a
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
- Every answer is tagged with the sources it actually drew on; each citation
  expands to the **exact retrieved snippet** it corresponds to - a specific
  document chunk or web result - not a generic "Uploaded source" label
- A **self-critique pass** re-checks the finished answer against that same
  context for claims it doesn't actually support, as an independent LLM call
  after the answer is generated - not the model grading its own homework in
  the same breath. Shown as a "✓ Claims checked" badge, or an expandable
  warning naming the specific unsupported claim(s) if the check fails
- Runs on Groq by default (fast, generous free tier); switch to a fully local
  model (e.g. Qwen2.5 via Ollama) or OpenAI with one environment variable -
  see [Choosing an LLM provider](#choosing-an-llm-provider)

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
5. **Critique** - once the answer is complete, a second, independent LLM call
   checks it against that same context for unsupported claims
6. **Cite** - the answer is returned with the specific sources - and exact
   snippets - it drew on, plus the critique verdict, so you can check its
   work instead of taking it on faith

![HonestRAG pipeline: user query flows through retrieve, concurrent relevance grading, a threshold decision that either trusts local context or triggers a Firecrawl web search, streamed answer generation, a self-critique pass, and a cited answer](assets/architecture.svg)

## Tech Stack

| Layer | Choice |
|---|---|
| RAG framework | [LlamaIndex](https://github.com/run-llama/llama_index) (Workflow-based orchestration) |
| LLM | [Groq](https://groq.com/) by default; [Ollama](https://ollama.com/) (local) or OpenAI via `LLM_PROVIDER` |
| Web search | [Firecrawl](https://firecrawl.dev/) |
| Vector store | [Milvus](https://milvus.io/) (Milvus Lite, local file-based) |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-large-en-v1.5`) |
| UI | [Streamlit](https://streamlit.io/) |

## Setup and Installation

### Prerequisites

- Python 3.11+
- A [Firecrawl](https://firecrawl.dev/) API key
- A [Groq](https://console.groq.com/) API key, **or** a locally running
  [Ollama](https://ollama.com/) instance if you'd rather not use an API key
  at all - see [Choosing an LLM provider](#choosing-an-llm-provider)

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

### Choosing an LLM provider

Both the answer-generation LLM and the relevance grader are built through a
single factory (`llm_provider.py`), switched with one variable:

```bash
LLM_PROVIDER=groq     # default - needs GROQ_API_KEY, GROQ_MODEL optional
LLM_PROVIDER=ollama   # fully local - needs a running `ollama serve`
LLM_PROVIDER=openai   # needs OPENAI_API_KEY, OPENAI_MODEL optional (default gpt-4o)
```

**Running fully local with Qwen (or any other Ollama model):**

```bash
ollama pull qwen2.5:7b        # or qwen2.5-coder:1.5b for a smaller/faster pull
```

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL="qwen2.5:7b"        # optional, defaults to qwen2.5-coder:1.5b
OLLAMA_BASE_URL="http://localhost:11434"  # optional, this is the default
```

No API key needed for this path - everything (retrieval grading, web-search
query rewriting, answer generation, self-critique) runs against your local
Ollama server instead. Expect it to be slower and less reliable at following
the grading/critique prompts than Groq's larger hosted models, especially
with a small model like `qwen2.5-coder:1.5b` - but it costs nothing and needs
no network access beyond Firecrawl's web-search fallback.

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
├── workflow.py             # CorrectiveRAGWorkflow: retrieve/grade/search/answer/critique
├── llm_provider.py          # Shared LLM factory (Groq / Ollama / OpenAI)
├── eval/
│   ├── dataset.jsonl        # Labeled examples for the relevance grader
│   └── run_eval.py          # Precision/recall/F1 harness for the grader
├── start_server.py          # Optional Beam Cloud deployment (unmodified from
│                             #   upstream; not part of the local Groq setup above)
├── requirements.txt
└── assets/architecture.svg   # Pipeline diagram used in this README
```

## Configuration

- **LLM provider**: `LLM_PROVIDER` in `.env` (`groq` / `ollama` / `openai`) -
  see [Choosing an LLM provider](#choosing-an-llm-provider); resolved once in
  `llm_provider.py` and used by both the UI and the workflow
- **Relevance threshold**: `CorrectiveRAGWorkflow.RELEVANCE_TRIGGER_THRESHOLD`
  in `workflow.py` (default `0.7`) - the minimum fraction of retrieved chunks
  that must grade "relevant" before local retrieval is trusted without a web
  search
- **Self-critique**: always on, adds one extra (non-streamed) LLM call after
  each answer; see `_critique_answer()` in `workflow.py` to disable or adjust
  its prompt
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

The grading LLM comes from the same `build_llm()` factory the app itself
uses (`llm_provider.py`), so the harness follows whatever `LLM_PROVIDER` is
already set in `.env` - the default is Groq, so in most cases no extra
credential is needed beyond what the app already requires:

```bash
python eval/run_eval.py
```

To evaluate against a fixed, independent baseline model instead of whatever
the app happens to be configured with:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY="your_openai_api_key_here" python eval/run_eval.py
```

`--dataset` points at a different labeled file if needed.

Sample run against the default Groq provider (`openai/gpt-oss-120b`), 20
examples: **accuracy 0.900, precision 0.833, recall 1.000, F1 0.909** (2
false positives, 0 false negatives) - the grader never let a truly relevant
chunk go unrecognized, but on 2 of the 20 examples called a topically-close
but non-answering chunk "relevant" when it wasn't.

### Mock mode

```bash
python eval/run_eval.py --mock
```

`--mock` (also triggered automatically if no credentials are found for the
active `LLM_PROVIDER`) swaps in a crude deterministic keyword-overlap
heuristic instead of a real LLM call. It exists solely so the harness's own
plumbing - dataset loading, prompt formatting, and metrics computation - can
be verified without an API key. Mock-mode numbers say nothing about the real
grader's quality; only a run with real provider credentials is a meaningful
evaluation.

## Acknowledgments

- [patchy631/ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub/tree/main/firecrawl-agent) for the original firecrawl-agent this was forked from
- [LlamaIndex](https://github.com/run-llama/llama_index) for the RAG framework
- [Firecrawl](https://firecrawl.dev/) for web search
- [Groq](https://groq.com/) for LLM inference
- [Milvus](https://milvus.io/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface
