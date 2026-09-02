from contextlib import redirect_stdout
import io
from workflow import CorrectiveRAGWorkflow, TokenEvent
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import uuid
import tempfile
import gc
import base64
import json
import qdrant_client
import streamlit as st
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

# The Milvus Lite database is a single local file, and the knowledge base
# manifest tracks which original filenames have been indexed into it. Both
# live next to the app so that the knowledge base survives across Streamlit
# restarts, not just across reruns within one session.
MILVUS_URI = "./milvus_demo.db"
MILVUS_COLLECTION = "firecrawl_agent_docs"
KB_MANIFEST_PATH = "./milvus_demo_docs.json"


# Set up page configuration
st.set_page_config(page_title="HonestRAG", layout="wide")


def load_indexed_docs():
    """Read the list of filenames already indexed into the persistent Milvus
    store, so the sidebar can show the real knowledge base contents even
    after an app restart."""
    if os.path.exists(KB_MANIFEST_PATH):
        try:
            with open(KB_MANIFEST_PATH, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
    return []


def save_indexed_docs(names):
    with open(KB_MANIFEST_PATH, "w") as f:
        json.dump(names, f)


def clear_knowledge_base():
    """Explicit, opt-in destructive reset. Wipes the persistent Milvus store
    and the filename manifest, and drops any workflow/index currently held in
    memory. This is the only place allowed to delete indexed documents."""
    if os.path.exists(MILVUS_URI):
        os.remove(MILVUS_URI)
    if os.path.exists(KB_MANIFEST_PATH):
        os.remove(KB_MANIFEST_PATH)

    st.session_state.workflow = None
    st.session_state.indexed_docs = []
    st.session_state.messages = []
    st.session_state.workflow_logs = []


# Initialize session state variables
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "indexed_docs" not in st.session_state:
    # Bootstrapped from disk so a restarted session immediately knows what's
    # already in the knowledge base, instead of assuming it's empty.
    st.session_state.indexed_docs = load_indexed_docs()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "workflow_logs" not in st.session_state:
    st.session_state.workflow_logs = []

session_id = st.session_state.id


@st.cache_resource
def load_llm():

    llm = OpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    return llm


def reset_chat():
    st.session_state.messages = []
    gc.collect()


def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Functions to build/update the workflow against the persistent Milvus store
#
# NOTE on `overwrite`: MilvusVectorStore only drops the collection when
# `overwrite=True` AND the collection already exists. When it doesn't exist
# yet, a fresh one is created regardless of `overwrite`. So passing
# `overwrite=False` here is enough to get "create on first use, reuse after
# that" for free - the collection, and therefore every previously uploaded
# document's embeddings, survives every subsequent call.
#
# NOTE on retrieval scope: `VectorStoreIndex.from_documents(...)` builds an
# `index_struct` that only contains the nodes passed to *that* call. Calling
# `.as_retriever()` on that specific index object restricts search to just
# those nodes (via a node_ids filter), which would silently limit answers to
# only the most recently uploaded document. To search across everything ever
# indexed, the workflow is handed an index built with
# `VectorStoreIndex.from_vector_store(...)` instead, which has an empty
# node_ids restriction and therefore queries the full Milvus collection.


def _build_settings_and_store():
    vector_store = MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name=MILVUS_COLLECTION,
        dim=1024,
        overwrite=False,
    )

    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5", cache_dir="./hf_cache")
    Settings.embed_model = embed_model

    llm = load_llm()
    Settings.llm = llm

    return vector_store, embed_model, llm


def _build_workflow(index, llm):
    if "FIRECRAWL_API_KEY" not in os.environ:
        raise ValueError("FireCrawl API key not found. Please enter it in the sidebar.")

    workflow = CorrectiveRAGWorkflow(
        index=index,
        firecrawl_api_key=os.environ["FIRECRAWL_API_KEY"],
        verbose=True,
        timeout=249,  # Increased timeout to match workflow execution
        llm=llm
    )
    print("DEBUG: Workflow created")
    return workflow


def add_documents_to_index(file_path, new_filenames):
    """Embed and insert the documents found under `file_path` into the
    persistent knowledge base, then rebuild the workflow so it can retrieve
    across every document indexed so far (not just this batch)."""
    try:
        with st.spinner("Loading documents and updating the knowledge base..."):
            documents = SimpleDirectoryReader(file_path).load_data()
            print(f"DEBUG: Loaded {len(documents)} documents")
            for i, doc in enumerate(documents):
                print(f"DEBUG: Document {i} preview: {doc.text[:100]}...")

            vector_store, embed_model, llm = _build_settings_and_store()
            print("DEBUG: Milvus vector store ready (persistent, not overwritten)")

            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            print("DEBUG: Storage context created")

            # Embeds and inserts the new documents' nodes into the existing
            # (or freshly created) Milvus collection. The returned index
            # object is intentionally not used for retrieval - see note above.
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
            )
            print("DEBUG: New documents inserted into the persistent index")

            # Rebuilt from the vector store so retrieval spans the whole
            # accumulated knowledge base, including documents from earlier
            # uploads/sessions.
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

            workflow = _build_workflow(index, llm)
            st.session_state.workflow = workflow

            for name in new_filenames:
                if name not in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.append(name)
            save_indexed_docs(st.session_state.indexed_docs)

            return workflow
    except Exception as e:
        st.error(f"Failed to update the knowledge base: {e}")
        raise e


def resume_workflow_from_existing_store():
    """Reconstruct a workflow from whatever is already persisted in Milvus,
    without requiring a new upload. Used to resume a session where the
    knowledge base already has documents from a previous run."""
    vector_store, embed_model, llm = _build_settings_and_store()
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    workflow = _build_workflow(index, llm)
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

# Sidebar for document upload
with st.sidebar:

    st.header("Add your documents!")

    uploaded_files = st.file_uploader(
        "Choose your `.pdf` file(s)", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        # The uploader re-sends every currently-selected file on each rerun,
        # so only process the ones not already in the knowledge base.
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_docs]

        if new_files:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for f in new_files:
                        file_path = os.path.join(temp_dir, f.name)
                        with open(file_path, "wb") as out_file:
                            out_file.write(f.getvalue())

                    st.write(f"Indexing {len(new_files)} new document(s)...")
                    add_documents_to_index(temp_dir, [f.name for f in new_files])

                st.success("Ready to Chat!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        else:
            st.success("Ready to Chat!")

        # Preview the most recently selected file.
        display_pdf(uploaded_files[-1])

    st.divider()
    st.subheader("Knowledge base")
    if st.session_state.indexed_docs:
        st.caption(f"{len(st.session_state.indexed_docs)} document(s) indexed")
        for name in st.session_state.indexed_docs:
            st.markdown(f"- {name}")
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.subheader("Danger zone")
    confirm_clear = st.checkbox("I understand this permanently deletes all indexed documents")
    if st.button("Clear knowledge base", disabled=not confirm_clear):
        clear_knowledge_base()
        st.success("Knowledge base cleared.")
        st.rerun()

# Main chat interface
col1, col2 = st.columns([6, 1])

with col1:
    # Centered main heading
    st.markdown('''
        <h1 style="text-align: center; font-weight: 500; color: #8de2ff;">
            HonestRAG
        </h1>
        <p style="text-align: center; color: #9aa5b1; margin-top: -8px;">
            Corrective RAG Agentic Workflow — retrieves, grades its own relevance, and falls back to live web search when local documents fall short.
        </p>
    ''', unsafe_allow_html=True)
    
    # Logos section below the heading
    st.markdown('''
        <div style="text-align: center; margin: 20px 0;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <img src="https://mintlify.s3.us-west-1.amazonaws.com/firecrawl/logo/logo-dark.png" alt="Firecrawl" style="height: 60px; margin-bottom: 5px;">
                </div>
                <div style="text-align: center;">
                    <img src="https://i.ibb.co/m5RtcvnY/beam-logo.png" alt="Beam Cloud" style="height: 60px; margin-bottom: 5px;">
                </div>
                <div style="text-align: center;">
                    <img src="https://milvus.io/images/layout/milvus-logo.svg" alt="Milvus" style="height: 60px; margin-bottom: 5px;">
                </div>
                <div style="text-align: center;">
                    <img src="https://www.comet.com/site/wp-content/uploads/2024/09/comet-logo-1.png" alt="CometML" style="height: 60px; margin-bottom: 5px;">
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Animation GIF section
    if "show_animation" not in st.session_state:
        st.session_state.show_animation = True
    
    if st.session_state.show_animation:
        st.image("https://d3e0luujhwn38u.cloudfront.net/original/img/original/186727/fbd774b8-29da-479a-a60c-880f84d66424.gif", use_container_width=True)

with col2:
    if st.button("Clear ↺", on_click=reset_chat):
        st.session_state.show_animation = False

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # If this is a user message and there are logs associated with it
    # Display logs AFTER the user message but BEFORE the next assistant message
    if message["role"] == "user" and "log_index" in message and i < len(st.session_state.messages) - 1:
        log_index = message["log_index"]
        if log_index < len(st.session_state.workflow_logs):
            with st.expander("View Workflow Execution Logs", expanded=False):
                st.code(
                    st.session_state.workflow_logs[log_index], language="text")

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
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
                elif hasattr(result, "response"):
                    answer_text = result.response
                    sources = []
                else:
                    answer_text = str(result)
                    sources = []

                # Prefer the structured answer; fall back to whatever was
                # streamed if for some reason the final text came back empty.
                full_response = answer_text or streamed_text["value"]
                message_placeholder.markdown(full_response)

                if sources:
                    source_lines = []
                    for source in sources:
                        if source.get("type") == "document":
                            source_lines.append("- Uploaded document")
                        elif source.get("type") == "web":
                            source_lines.append(f"- Web: {source.get('url')}")
                    if source_lines:
                        with st.expander("Sources used", expanded=False):
                            st.markdown("\n".join(source_lines))

            # Display the workflow logs in an expandable section AFTER the
            # assistant chat bubble (moved from before it: the logs aren't
            # captured until run_workflow finishes, but the chat bubble now
            # has to exist beforehand so tokens can stream into it live).
            if log_index < len(st.session_state.workflow_logs):
                with st.expander("View Workflow Execution Logs", expanded=False):
                    st.code(
                        st.session_state.workflow_logs[log_index], language="text")

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
