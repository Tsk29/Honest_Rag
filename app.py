from contextlib import redirect_stdout
import io
from workflow import TokenEvent
from llama_index.core import Settings
from llm_provider import build_llm
import knowledge_base as kb
import rag_service
import uuid
import tempfile
import gc
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# Where the vectors live (a local Milvus Lite file, or a hosted Milvus such as
# Zilliz Cloud) is configured by HONESTRAG_MILVUS_URI / HONESTRAG_MILVUS_TOKEN - see
# knowledge_base.py. Indexing and workflow construction are shared with the
# Gradio app - see rag_service.py.


# Set up page configuration
st.set_page_config(page_title="HonestRAG", layout="wide", page_icon="📓")

# NotebookLM-inspired theme: dark charcoal canvas, a narrow "Sources" rail on
# the left, flat rounded cards instead of Streamlit's default boxy widgets,
# and a restrained blue accent. This only reskins existing Streamlit
# components via their stable data-testid hooks - no layout logic changes.
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600&family=Google+Sans+Text&display=swap" rel="stylesheet">
<style>
    :root {
        --nb-bg: #131314;
        --nb-surface: #1e1f20;
        --nb-surface-hover: #26282a;
        --nb-border: #3c4043;
        --nb-text: #e3e3e3;
        --nb-text-dim: #9aa0a6;
        --nb-accent: #a8c7fa;
    }

    html, body, [class*="css"] { font-family: "Google Sans Text", Roboto, "Segoe UI", system-ui, sans-serif; }
    .stApp { background: var(--nb-bg); }

    /* Center the workspace in a comfortable reading width instead of
       stretching chat bubbles across the full wide-layout viewport. */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar -> Sources rail */
    section[data-testid="stSidebar"] {
        background: var(--nb-surface);
        border-right: 1px solid var(--nb-border);
        width: 320px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.25rem 1rem;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: var(--nb-text);
        font-weight: 500;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        opacity: 0.75;
        margin-bottom: 0.75rem;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem;
    }

    /* Upload dropzone as a rounded, dashed "add source" card */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--nb-bg) !important;
        border: 1.5px dashed var(--nb-border) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--nb-accent) !important;
    }

    /* Buttons - pill-shaped, flat */
    .stButton > button {
        border-radius: 999px;
        border: 1px solid var(--nb-border);
        background: var(--nb-surface);
        color: var(--nb-text);
    }
    .stButton > button:hover {
        border-color: var(--nb-accent);
        color: var(--nb-accent);
    }

    /* Source cards inside the Sources list */
    .nb-source-card {
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--nb-bg);
        border: 1px solid var(--nb-border);
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 0.85rem;
        color: var(--nb-text);
        height: 100%;
        box-sizing: border-box;
    }
    .nb-source-card .nb-icon { opacity: 0.8; }
    .nb-source-info {
        display: flex;
        flex-direction: column;
        gap: 1px;
        min-width: 0;
    }
    .nb-source-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .nb-source-meta {
        font-size: 0.72rem;
        color: var(--nb-text-dim);
    }

    /* Per-source remove ("x") button: compact and quiet, not a full pill */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        align-items: stretch;
        gap: 4px;
        margin-bottom: 6px;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button {
        width: 100%;
        height: 100%;
        min-height: 0;
        padding: 0;
        color: var(--nb-text-dim);
        font-size: 1rem;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
        color: #f28b82;
        border-color: #f28b82;
    }

    /* Chat input as a floating rounded bar */
    [data-testid="stChatInput"] textarea {
        border-radius: 24px !important;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: var(--nb-surface);
        border-radius: 16px;
        border: 1px solid var(--nb-border);
        padding: 4px 8px;
    }

    /* Expanders (sources / logs) as subtle chips */
    [data-testid="stExpander"] {
        border: 1px solid var(--nb-border) !important;
        border-radius: 12px !important;
        background: var(--nb-surface) !important;
    }

    /* Empty-state welcome card */
    .nb-empty-state {
        max-width: 560px;
        margin: 8vh auto 0 auto;
        text-align: center;
        padding: 32px;
        border: 1px solid var(--nb-border);
        border-radius: 20px;
        background: var(--nb-surface);
    }
    .nb-empty-state h2 {
        color: var(--nb-text);
        font-weight: 500;
        margin-bottom: 8px;
    }
    .nb-empty-state p {
        color: var(--nb-text-dim);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Top bar */
    .nb-topbar {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 18px 0;
        border-bottom: 1px solid var(--nb-border);
        margin-bottom: 20px;
    }
    .nb-topbar .nb-title {
        font-size: 1.15rem;
        font-weight: 500;
        color: var(--nb-text);
    }
    .nb-topbar .nb-badge {
        font-size: 0.75rem;
        color: var(--nb-text-dim);
        border: 1px solid var(--nb-border);
        border-radius: 999px;
        padding: 2px 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_kb_client():
    return kb.make_client()


def refresh_indexed_docs():
    """Re-read the source list (name, size, page count) from the vector
    store itself - the single source of truth - so the sidebar reflects what
    is really indexed, including documents from earlier runs."""
    st.session_state.indexed_docs = kb.list_sources(get_kb_client())


def clear_knowledge_base():
    """Explicit, opt-in destructive reset. Drops the whole collection and any
    workflow/index currently held in memory. This is the only place allowed
    to delete every indexed document."""
    kb.clear(get_kb_client())

    st.session_state.workflow = None
    st.session_state.indexed_docs = []
    st.session_state.messages = []
    st.session_state.workflow_logs = []
    st.session_state.uploader_key += 1


# Initialize session state variables
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "indexed_docs" not in st.session_state:
    # Bootstrapped from the vector store so a restarted session immediately
    # knows what's already in the knowledge base, instead of assuming it's
    # empty.
    try:
        refresh_indexed_docs()
    except Exception as e:
        st.session_state.indexed_docs = []
        st.warning(f"Could not connect to the knowledge base: {e}")

if "uploader_key" not in st.session_state:
    # Bumped whenever a source is removed (individually or via "Clear
    # knowledge base") so the file_uploader widget below is recreated with a
    # fresh key. Streamlit's uploader otherwise keeps holding the browser's
    # file selection across reruns - without this, a just-removed file would
    # still show up in `uploaded_files` on the very next rerun and get
    # silently re-indexed right back in.
    st.session_state.uploader_key = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "workflow_logs" not in st.session_state:
    st.session_state.workflow_logs = []

session_id = st.session_state.id


@st.cache_resource
def load_llm():
    # Provider is picked by LLM_PROVIDER ("groq" default, "ollama" for a
    # fully local model such as Qwen2.5, "openai") - see llm_provider.py.
    return build_llm()


def reset_chat():
    st.session_state.messages = []
    gc.collect()


@st.cache_resource
def load_embed_model():
    # Loading the ~1.3 GB embedding model is slow; do it once per process,
    # not on every upload.
    return rag_service.build_embed_model()


def _build_settings_and_store():
    vector_store = rag_service.open_vector_store()

    embed_model = load_embed_model()
    Settings.embed_model = embed_model

    llm = load_llm()
    Settings.llm = llm

    return vector_store, embed_model, llm


def add_documents_to_index(file_path):
    """Embed and insert the documents found under `file_path` into the
    persistent knowledge base, then rebuild the workflow so it can retrieve
    across every document indexed so far (not just this batch).
    """
    try:
        with st.spinner("Loading documents and updating the knowledge base..."):
            documents = rag_service.load_documents(file_path)
            print(f"DEBUG: Loaded {len(documents)} documents")

            vector_store, embed_model, llm = _build_settings_and_store()
            rag_service.index_documents(documents, vector_store, embed_model)
            print("DEBUG: New documents inserted into the persistent index")

            workflow = rag_service.build_workflow(vector_store, embed_model, llm)
            st.session_state.workflow = workflow

            refresh_indexed_docs()

            return workflow
    except Exception as e:
        st.error(f"Failed to update the knowledge base: {e}")
        raise e


def remove_document(name: str):
    """Remove a single source from the persistent knowledge base without
    touching any other indexed document, using the stable ref_doc_id
    (the original filename) `add_documents_to_index` assigns at index time.
    """
    vector_store, embed_model, llm = _build_settings_and_store()
    rag_service.remove_source(vector_store, name)

    refresh_indexed_docs()

    # Force the file_uploader to remount empty - see the uploader_key
    # comment above - otherwise the just-removed file is still sitting in
    # the widget's selection and gets treated as "new" on the next rerun.
    st.session_state.uploader_key += 1

    if st.session_state.indexed_docs:
        try:
            st.session_state.workflow = rag_service.build_workflow(vector_store, embed_model, llm)
        except Exception as e:
            st.session_state.workflow = None
            st.sidebar.warning(f"Could not rebuild the workflow after removing a source: {e}")
    else:
        st.session_state.workflow = None


def resume_workflow_from_existing_store():
    """Reconstruct a workflow from whatever is already persisted in Milvus,
    without requiring a new upload. Used to resume a session where the
    knowledge base already has documents from a previous run."""
    vector_store, embed_model, llm = _build_settings_and_store()
    workflow = rag_service.build_workflow(vector_store, embed_model, llm)
    st.session_state.workflow = workflow
    return workflow


# Resume automatically if the knowledge base already has documents (e.g. from
# an earlier run of the app) but this session hasn't built a workflow yet.
# Best-effort: without a FireCrawl key configured (as in this dev
# environment), this just leaves the workflow unset instead of crashing the
# whole app - the user still sees the existing document list in the sidebar.
if st.session_state.workflow is None and st.session_state.indexed_docs:
    try:
        resume_workflow_from_existing_store()
    except Exception as e:
        st.sidebar.warning(f"Could not resume the existing knowledge base: {e}")

# Function to run the async workflow


async def run_workflow(query, on_token=None):
    """Run the workflow to completion, optionally streaming tokens as they arrive.

    `CorrectiveRAGWorkflow.run(...)` returns a `WorkflowHandler` immediately
    (it doesn't need to be awaited to start the run). That handler is both
    awaitable (for the final `StopEvent` result) and exposes
    `stream_events()`, an async generator of every event a step writes via
    `ctx.write_event_to_stream()` - including the `TokenEvent`s that
    `query_result()` emits for each streamed LLM delta. Iterating that
    generator to completion is what actually drives/advances the workflow;
    it ends automatically once the StopEvent is produced, at which point
    `await handler` resolves instantly with the final result.

    `on_token`, if given, is called synchronously with each token's delta
    string as it streams in - the caller (see the chat-input handling below)
    uses it to update the Streamlit placeholder live, instead of faking a
    reveal after the fact.
    """
    try:
        # Capture stdout to get the workflow logs
        f = io.StringIO()
        with redirect_stdout(f):
            async def _drive_and_collect():
                handler = st.session_state.workflow.run(query_str=query)
                async for event in handler.stream_events():
                    if isinstance(event, TokenEvent) and on_token is not None:
                        on_token(event.delta)
                return await handler

            # Add timeout to prevent hanging
            result = await asyncio.wait_for(_drive_and_collect(), timeout=120)

        # Get the captured logs and store them
        logs = f.getvalue()
        if logs:
            st.session_state.workflow_logs.append(logs)

        return result
    except asyncio.TimeoutError:
        st.error("Workflow execution timed out after 2 minutes")
        raise Exception("Workflow execution timed out")
    except Exception as e:
        # Log the error and re-raise it
        st.error(f"Workflow execution failed: {e}")
        raise e

# Sidebar as a NotebookLM-style "Sources" rail
with st.sidebar:

    st.header("📓 Sources")

    uploaded_files = st.file_uploader(
        "Add source", type="pdf", accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        # The uploader re-sends every currently-selected file on each rerun,
        # so only process the ones not already in the knowledge base.
        existing_names = {d["name"] for d in st.session_state.indexed_docs}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        if new_files:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for f in new_files:
                        file_path = os.path.join(temp_dir, f.name)
                        with open(file_path, "wb") as out_file:
                            out_file.write(f.getvalue())

                    st.caption(f"Indexing {len(new_files)} new source(s)...")
                    add_documents_to_index(temp_dir)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

    st.markdown("&nbsp;", unsafe_allow_html=True)

    if st.session_state.indexed_docs:
        st.caption(f"{len(st.session_state.indexed_docs)} source(s)")
        for doc in st.session_state.indexed_docs:
            meta_bits = []
            if doc.get("size_kb"):
                meta_bits.append(f'{doc["size_kb"]} KB')
            if doc.get("pages"):
                meta_bits.append(f'{doc["pages"]} page{"s" if doc["pages"] != 1 else ""}')
            meta_text = " · ".join(meta_bits)

            row_col, remove_col = st.columns([5, 1])
            with row_col:
                st.markdown(
                    f'<div class="nb-source-card">'
                    f'<span class="nb-icon">📄</span>'
                    f'<span class="nb-source-info">'
                    f'<span class="nb-source-name">{doc["name"]}</span>'
                    + (f'<span class="nb-source-meta">{meta_text}</span>' if meta_text else '')
                    + '</span></div>',
                    unsafe_allow_html=True,
                )
            with remove_col:
                if st.button("×", key=f"remove_{doc['name']}", help=f"Remove {doc['name']}"):
                    remove_document(doc["name"])
                    st.rerun()
    else:
        st.caption("No sources yet. Add a PDF above to get started.")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Danger zone"):
        confirm_clear = st.checkbox("I understand this permanently deletes all indexed documents")
        if st.button("Clear knowledge base", disabled=not confirm_clear):
            clear_knowledge_base()
            st.success("Knowledge base cleared.")
            st.rerun()

# Top bar: compact title + source count badge + clear-chat control. No
# framework logos or decorative animation - the landing view goes straight
# to the workspace, the way NotebookLM's does.
topbar_col, clear_col = st.columns([6, 1])
with topbar_col:
    n_sources = len(st.session_state.indexed_docs)
    badge_text = f"{n_sources} source{'s' if n_sources != 1 else ''}"
    st.markdown(f'''
        <div class="nb-topbar">
            <span style="font-size: 1.4rem;">📓</span>
            <span class="nb-title">HonestRAG</span>
            <span class="nb-badge">{badge_text}</span>
        </div>
    ''', unsafe_allow_html=True)
with clear_col:
    st.button("Clear ↺", on_click=reset_chat)

# Empty state: shown until the first message is sent, mirroring NotebookLM's
# "ask something about your sources" landing prompt instead of a wall of logos.
# Held in a placeholder so it can be cleared immediately below once a message
# is appended in this same script run - otherwise it would linger for one
# extra render, since the message list only reflects the new turn from here on.
empty_state = st.empty()
if not st.session_state.messages:
    empty_state.markdown('''
        <div class="nb-empty-state">
            <h2>Ask HonestRAG anything</h2>
            <p>
                It answers from your uploaded sources first, grades how relevant
                what it found actually is, and automatically falls back to a live
                web search when your sources don't cover the question.
            </p>
        </div>
    ''', unsafe_allow_html=True)

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    empty_state.empty()

    # Add user message to chat history with placeholder for log index
    log_index = len(st.session_state.workflow_logs)
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "log_index": log_index})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = ""

    if st.session_state.workflow:
        try:
            # Display assistant response in chat message container. The
            # placeholder is created BEFORE run_workflow is called so the
            # on_token callback below can update it live, token by token, as
            # the workflow's stream_events() delivers them - Streamlit
            # renders each .markdown() call as it happens even though we're
            # still inside asyncio.run(), which is what makes this a real
            # reveal instead of a post-hoc fake one.
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                streamed_text = {"value": ""}

                def _on_token(delta: str):
                    streamed_text["value"] += delta
                    message_placeholder.markdown(streamed_text["value"] + "▌")

                # Run the async workflow with proper error handling
                result = asyncio.run(run_workflow(prompt, on_token=_on_token))

                if isinstance(result, dict):
                    answer_text = result.get("answer", "")
                    sources = result.get("sources", [])
                    critique = result.get("critique")
                elif hasattr(result, "response"):
                    answer_text = result.response
                    sources = []
                    critique = None
                else:
                    answer_text = str(result)
                    sources = []
                    critique = None

                # Prefer the structured answer; fall back to whatever was
                # streamed if for some reason the final text came back empty.
                full_response = answer_text or streamed_text["value"]
                message_placeholder.markdown(full_response)

                # Each source is shown as its exact retrieved snippet, not a
                # generic "Uploaded source" label - collapsed by default so
                # the answer stays the focus, but one click away from
                # checking precisely what backed it.
                if sources:
                    doc_n = 0
                    for source in sources:
                        if source.get("type") == "document":
                            doc_n += 1
                            label = f"📄 Uploaded source {doc_n}" if len([s for s in sources if s.get("type") == "document"]) > 1 else "📄 Uploaded source"
                        else:
                            label = f'🌐 {source.get("url")}'
                        with st.expander(label, expanded=False):
                            st.caption(source.get("text", ""))

                # Self-critique verdict: a second, independent LLM pass that
                # checked the finished answer against the same context for
                # unsupported claims. Silent when it couldn't run at all
                # (e.g. transient API error) rather than implying a result
                # that was never actually computed.
                if critique is not None:
                    if critique.get("passed"):
                        st.caption("✓ Claims checked against sources")
                    else:
                        with st.expander("⚠️ Some claims may not be fully supported by sources", expanded=False):
                            st.caption(critique.get("note", ""))

        except Exception as e:
            st.error(f"Error running workflow: {e}")
            full_response = f"An error occurred while processing your request: {e}"
            st.markdown(full_response)
        # else:
        #     full_response = "Please upload a document first to initialize the workflow."
        #     st.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
