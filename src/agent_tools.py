import re
import sys
import time
import requests
from pathlib import Path
from typing import Any
from sqlalchemy import Engine, text

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ==========================================
# 1. ENVIRONMENT & DYNAMIC INDEX ATTACHMENT
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import EmbeddingsModel, reveal, settings
from src.db import make_engine


def get_cached_huggingface_embeddings(model_name: str) -> Embeddings:
    """
    Loads and locks the HuggingFace model weights into the machine's global RAM.
    If called again during any subsequent script rerun, it returns instantly.
    """
    import streamlit as st
    
    # We wrap the inner call with st.cache_resource dynamically 
    @st.cache_resource(show_spinner=False)
    def _load_model(name: str) -> Embeddings:
        print(f"🧠 MEMORY SEED: Permanently caching local model [{name}] in global RAM...")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={'device': 'cpu'}
        )
    return _load_model(model_name)

EMBEDDINGS_MODEL_SETTING = settings.embeddings_model

db_user = settings.sql_agent_user
db_password = reveal(settings.sql_agent_password)

INDEX_NAME = "fde-sop-index-openai" if EMBEDDINGS_MODEL_SETTING is EmbeddingsModel.OPENAI else "fde-sop-index-local"

if EMBEDDINGS_MODEL_SETTING is EmbeddingsModel.OPENAI:
    print("🤖 Mode: Connecting to Cloud OpenAI Index (1536 Dim Space)...")
    embeddings = OpenAIEmbeddings()
else :
    local_model_target = settings.local_embedding_model
    
    print(f"🤗 Mode: Connecting to Local Fallback [{local_model_target}] Index (1024 Dim Space)...")

    try:
        from streamlit.runtime import exists as streamlit_runtime_exists
        if streamlit_runtime_exists():
            embeddings = get_cached_huggingface_embeddings(local_model_target)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=local_model_target, model_kwargs={'device': 'cpu'})
    except ImportError:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=local_model_target, model_kwargs={'device': 'cpu'})

vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ==========================================
# 2. CORE FDE AGENT TOOLS
# ==========================================

# Keep tool output small: it is re-read by a (slow) LLM on every following step.
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_QUERY_ROWS = 10

_telemetry_engine: Engine | None = None


def get_telemetry_engine() -> Engine:
    """Build the read-only engine once and reuse its connection pool across tool calls."""
    global _telemetry_engine
    if _telemetry_engine is None:
        _telemetry_engine = make_engine(db_user, db_password)
    return _telemetry_engine


# Defence in depth on top of the read-only DB login: a single plain SELECT only.
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|EXEC|EXECUTE|GRANT|REVOKE|INTO|OPENROWSET|OPENQUERY|XP_\w+|SP_\w+)\b",
    re.IGNORECASE,
)


def validate_select_only(sql_query: str) -> str | None:
    """Return an error message if the query isn't a single plain SELECT, else None."""
    stripped = sql_query.strip().rstrip(";").strip()
    if not stripped.upper().startswith("SELECT"):
        return "SECURITY BLOCK: Only SELECT operations are authorized on this view."
    if ";" in stripped or "--" in stripped or "/*" in stripped:
        return "SECURITY BLOCK: Multiple statements and SQL comments are not allowed."
    if _FORBIDDEN_SQL.search(stripped):
        return "SECURITY BLOCK: Query contains a forbidden keyword."
    return None


def _truncate(text_value: str) -> str:
    if len(text_value) <= MAX_TOOL_OUTPUT_CHARS:
        return text_value
    return text_value[:MAX_TOOL_OUTPUT_CHARS] + "\n...[output truncated]"


@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a SQL SELECT query against the FDE_VIEWS.VW_ACTIVE_FLEET view.
    Columns available:
    Timestamp, Latitude, Longitude, Current_Temperature_C, Cargo_Condition_Code,
    Risk_Classification, Delay_Probability, Port_Congestion_Level, Route_Risk_Index.
    Always write standard T-SQL queries.
    """
    blocked = validate_select_only(sql_query)
    if blocked:
        return blocked

    try:
        with get_telemetry_engine().connect() as conn:
            cursor = conn.execute(text(sql_query))
            columns = list(cursor.keys())
            rows = cursor.fetchmany(MAX_QUERY_ROWS)

            if not rows:
                return "No records matched the query criteria."

            lines = [f"COLUMNS: {', '.join(columns)}"]
            lines.extend(str(tuple(row)) for row in rows)
            return _truncate("\n".join(lines) + "\n")
    except Exception as e:
        return f"Database Error: {str(e)}"


_http = requests.Session()
_CORRIDOR_TTL_SECONDS = 600
_corridor_cache: dict[tuple[float, float], tuple[float, str]] = {}


@tool
def fetch_corridor_conditions(latitude: float, longitude: float) -> str:
    """
    Fetches real-time weather and corridor conditions from a live REST API for given GPS coordinates.
    Provides temperature, wind speed, and computed corridor congestion index.
    """
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return "Invalid coordinates: latitude must be within -90..90 and longitude within -180..180."

    # Weather barely changes in minutes; reuse a recent answer for the same ~1 km cell.
    cache_key = (round(latitude, 2), round(longitude, 2))
    cached = _corridor_cache.get(cache_key)
    if cached is not None and time.monotonic() - cached[0] < _CORRIDOR_TTL_SECONDS:
        return cached[1]

    try:
        response = _http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude, "current_weather": "true"},
            timeout=6,
        )
        response.raise_for_status()

        payload: dict[str, Any] = response.json().get("current_weather", {})
        temp = payload.get("temperature", "N/A")
        wind = payload.get("windspeed", 0.0)

        congestion_index = 8.5 if wind > 10.0 else 2.5
        status_note = "High Transit Disruption" if wind > 10.0 else "Corridor Normal"

        result = (
            f"--- LIVE CORRIDOR TELEMETRY ---\n"
            f"Target GPS: {latitude}, {longitude}\n"
            f"External Temp: {temp}°C | Wind Speed: {wind} km/h\n"
            f"Corridor Risk: {status_note} (Congestion Index: {congestion_index}/10)\n"
            f"-------------------------------"
        )
        if len(_corridor_cache) >= 256:  # bound memory
            _corridor_cache.clear()
        _corridor_cache[cache_key] = (time.monotonic(), result)
        return result
    except Exception as e:
        return f"Corridor API Communication Failure: {str(e)}"

@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches enterprise Standard Operating Procedures (SOPs) indexed in the Pinecone Vector DB.
    Use this to retrieve regulatory thresholds, cold-chain breach mitigations, and rerouting rules.
    """
    try:
        matched_docs = retriever.invoke(query)
        if not matched_docs:
            return "No matching compliance clauses found."
            
        formatted_context = "\n\n".join(
            [f"[Source: {doc.metadata.get('source_file', 'SOP')} | Format: {doc.metadata.get('file_format', 'RAW')}]\n{doc.page_content}" for doc in matched_docs]
        )
        return f"--- COMPLIANCE SOP CONTEXT ---\n{formatted_context}\n------------------------------"
    except Exception as e:
        return f"Vector Store Retrieval Error: {str(e)}"

# ==========================================
# 3. LOCAL VERIFICATION
# ==========================================
if __name__ == "__main__":
    print("\n--- Testing Tool 1: SQL Telemetry View ---")
    print(query_telemetry_db.invoke("SELECT TOP 2 Latitude, Longitude, Current_Temperature_C FROM FDE_VIEWS.VW_ACTIVE_FLEET"))
    
    print("\n--- Testing Tool 2: Live Corridor API ---")
    print(fetch_corridor_conditions.invoke({"latitude": 33.77, "longitude": -118.19}))
    
    print("\n--- Testing Tool 3: Pinecone Vector Retrieval ---")
    print(search_compliance_sop.invoke("What are the temperature rules for fresh perishables?"))
