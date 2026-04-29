from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated, Any, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.utilities.requests import GenericRequestsWrapper
from langchain_core.tools import tool
import requests
import datetime
import tempfile
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05
)

model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#PDF retriever store (per thread)

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

#tools

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

# Custom datetime tool
@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    return str(datetime.datetime.now())

# Custom weather tool (free, no API key needed)
@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city name."""
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        ).json()
        if not geo.get("results"):
            return f"City not found: {city}"
        r = geo["results"][0]
        lat, lon = r["latitude"], r["longitude"]
        name, country = r["name"], r.get("country", "")
        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,windspeed_10m,relative_humidity_2m"
            f"&temperature_unit=celsius"
        ).json()["current"]
        return (
            f"Weather in {name}, {country}:\n"
            f"🌡️ Temperature: {weather['temperature_2m']}°C\n"
            f"💨 Wind Speed: {weather['windspeed_10m']} km/h\n"
            f"💧 Humidity: {weather['relative_humidity_2m']}%"
        )
    except Exception as e:
        return f"Weather fetch failed: {str(e)}"

# Custom calculator tool (free, no API key needed)
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '2 + 2' or '15% of 3840'."""
    try:
        if "% of" in expression:
            parts = expression.replace("%", "").split("of")
            result = (float(parts[0].strip()) / 100) * float(parts[1].strip())
            return f"Result: {result}"
        safe_builtins = {k: v for k, v in vars(__builtins__).items()
                         if k in ['abs', 'round', 'min', 'max', 'pow', 'sum']}
        result = eval(expression, {"__builtins__": safe_builtins})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation failed: {str(e)}"

search_tool = DuckDuckGoSearchRun(region="us-en")
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
requests_wrapper = GenericRequestsWrapper()
web_scraper_tool = RequestsGetTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True)

tools = [rag_tool, search_tool, wikipedia_tool, get_weather, calculator, web_scraper_tool, get_current_datetime]
llm_with_tools = model.bind_tools(tools)

#state definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#chat node definition
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
    content=(
        "You are a helpful assistant. If the user asks anything about an uploaded file, "
        "document, PDF, DOCX, TXT, notes, report, or its contents (examples: what is this document about, summarize this file, explain this report), "
        "always call the `rag_tool` and include the thread_id "
        f"`{thread_id}`. You can also use web search tools when helpful. "
        "If no document is available, ask the user to upload a file."
    )
)
        

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

#checkpointer to save the state of the conversation in memory
checkpointer = MemorySaver()


#graph
graph= StateGraph(ChatState)

#nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
#edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot= graph.compile(checkpointer=checkpointer)

# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})