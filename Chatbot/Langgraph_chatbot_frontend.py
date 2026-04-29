import streamlit as st
from chat_backend import (
    chatbot,
    model,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import docx

# **************************************** Utility Functions *************************

def generate_thread_id():
    return str(uuid.uuid4())  # always str to avoid UUID vs str key mismatch

def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
    add_thread(st.session_state['thread_id'])

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def generate_chat_title(user_message: str, ai_response: str) -> str:
    """Use a cheap LLM call to generate a concise 3-5 word chat title."""
    prompt = (
        f"User: {user_message}\nAssistant: {ai_response}\n\n"
        "Give a concise 3-5 word title for this conversation. "
        "Reply with the title only, no punctuation."
    )
    response = model.invoke(prompt)
    return response.content.strip()


# **************************************** Session Setup *****************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()  # load persisted threads

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = {}  # str(thread_id) -> {filename -> summary}

if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = {}  # str(thread_id) -> title

add_thread(st.session_state['thread_id'])

# Compute thread_key once so it's consistent everywhere (fixes UUID vs str mismatch)
thread_key = str(st.session_state['thread_id'])
thread_docs = st.session_state['ingested_docs'].setdefault(thread_key, {})
selected_thread = None


# **************************************** Sidebar UI ********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat 📝', use_container_width=True):
    reset_chat()
    st.rerun()

# Show currently indexed PDF for this thread (if any)
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📄 Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

# PDF upload lives in sidebar so it triggers independently of chat input
uploaded_pdf = st.sidebar.file_uploader("Upload your PDF and question answer with the chatbot", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.header('Chat History')

for thread_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state['chat_titles'].get(str(thread_id), str(thread_id)[:8] + '...')
    if st.sidebar.button(title, key=f'thread_{thread_id}'):
        selected_thread = thread_id  # defer switch to bottom to avoid mid-render mutation


# **************************************** Main UI ***********************************

if not st.session_state['message_history']:
    st.title("What do you want to talk about?")
    st.markdown("Here are some things I can help you with:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✍️ Help me write something", use_container_width=True):
            st.session_state['suggested_prompt'] = "Help me write something"
        if st.button("📝Todays Latest News", use_container_width=True):
            st.session_state['suggested_prompt'] = "Help me list and describe today's latest news"
        if st.button("📚 Search Wikipedia", use_container_width=True):
            st.session_state['suggested_prompt'] = "Search Wikipedia for me"

    with col2:
        if st.button("💡 Brainstorm ideas", use_container_width=True):
            st.session_state['suggested_prompt'] = "Help me brainstorm some ideas"
        if st.button("📖 Explain a concept", use_container_width=True):
            st.session_state['suggested_prompt'] = "Explain a concept to me"
        if st.button("🧮 Calculator", use_container_width=True):
            st.session_state['suggested_prompt'] = "Calculate mathematical problem for me"

    with col3:
        if st.button("📝 Summarize a text", use_container_width=True):
            st.session_state['suggested_prompt'] = "Summarize a text for me"
        if st.button("🌐 Translate something", use_container_width=True):
            st.session_state['suggested_prompt'] = "Translate something for me"
        if st.button("🔗 Summarize a URL", use_container_width=True):
            st.session_state['suggested_prompt'] = "Fetch and summarize this URL: https://example.com"

avatars = {'user': '👨‍💻', 'assistant': '✨'}

for message in st.session_state['message_history']:
    with st.chat_message(message['role'], avatar=avatars[message['role']]):
        st.markdown(message['content'])  # markdown instead of st.text so LLM formatting renders

st.markdown("""
    <style>
        .upload-btn {
            position: fixed;
            bottom: 22px;
            left: 80px;
            z-index: 999;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
    with st.popover("📎"):
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=["txt", "docx"],  # PDF now handled in sidebar; popover for TXT/DOCX only
            label_visibility="collapsed"
        )
st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input('Type here......') or st.session_state.pop('suggested_prompt', None)

if user_input:

    # Extract file content if uploaded (TXT / DOCX only — PDF is handled in sidebar)
    file_context = ""
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            file_context = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            file_context = "\n".join(para.text for para in doc.paragraphs if para.text)

    # Append file content to the user message so the LLM actually sees it
    full_message = f"{user_input}\n\n{file_context}" if file_context else user_input

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user', avatar='👨‍💻'):
        st.markdown(user_input)

    CONFIG = {
        'configurable': {'thread_id': thread_key},
        "metadata": {
            "thread_id": thread_key,
        },
        "run_name": "chat_run",
    }

    with st.chat_message('assistant', avatar='✨'):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=full_message)]},
                config=CONFIG,
                stream_mode='messages'
            ):
                # Show which tool is being called in a status widget
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, 'name', 'tool')
                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder['box'].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder['box'] is not None:
            status_holder['box'].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Show indexed document metadata below the response if available
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

    # Generate and save title after the first exchange in a new thread
    if thread_key not in st.session_state['chat_titles']:
        title = generate_chat_title(user_input, ai_message)
        st.session_state['chat_titles'][thread_key] = title
        st.rerun()

# ---- Deferred thread switch (avoids mid-render state mutation) ----
if selected_thread:
    st.session_state['thread_id'] = str(selected_thread)
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        temp_messages.append({'role': role, 'content': msg.content})

    st.session_state['message_history'] = temp_messages
    st.session_state['ingested_docs'].setdefault(str(selected_thread), {})
    st.rerun()