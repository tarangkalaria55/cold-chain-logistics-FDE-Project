import sys
import uuid
import json
import hmac
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import pandas as pd
from sqlalchemy import Engine, text
from langchain_core.messages import HumanMessage, ToolMessage

# ==========================================
# 1. IMMEDIATE PATH & ENVIRONMENT RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent  # points to src/
project_root = script_dir.parent              # climbs to project root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the compiled graph and tools list dynamically
from src.config import reveal, settings
from src.db import make_engine
from src.orchestrator import AgentState, checkpointer, fde_agent, thread_config_for

# ==========================================
# 2. SQL CREDENTIALS MAPPING FROM .ENV
# ==========================================
db_user = settings.sql_agent_user
db_password = reveal(settings.sql_agent_password)

MAX_ADMIN_LOGIN_FAILURES = 5


@st.cache_resource(show_spinner=False)
def get_log_engine() -> Engine:
    """One pooled engine for the whole server; Streamlit reruns the script on every click."""
    return make_engine(db_user, db_password)


@st.cache_resource(show_spinner=False)
def get_audit_executor() -> ThreadPoolExecutor:
    """Single background worker so audit inserts never block the streaming UI (order is kept)."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit-log")


def _write_audit_log_now(engine: Engine, session_id: str, node_name: str, tool_name: str | None, content: str) -> None:
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO FDE_VIEWS.AgentAuditLog (SessionID, NodeExecuted, ToolName, Content)
                VALUES (:session_id, :node_name, :tool_name, :content)
            """), {
                "session_id": session_id,
                "node_name": node_name,
                "tool_name": tool_name,
                "content": content
            })
            conn.commit()
    except Exception as e:
        print(f"Audit Log Failed (Silent): {e}")


def write_audit_log(session_id: str, node_name: str, tool_name: str | None, content: Any) -> None:
    """Queues an agent execution trace for the SQL audit table (agent permissions, non-blocking)."""
    text_content = content if isinstance(content, str) else json.dumps(content, default=str)
    try:
        get_audit_executor().submit(_write_audit_log_now, get_log_engine(), session_id, node_name, tool_name, text_content)
    except Exception as e:
        print(f"Audit Log Failed (Silent): {e}")

# ==========================================
# 3. PAGE CONFIGURATION (theme lives in .streamlit/config.toml)
# ==========================================
st.set_page_config(
    page_title="FDE dispatch console",
    page_icon=":material/ac_unit:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 4. MULTI-USER STATE & THREAD MANAGEMENT
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

if "admin_login_failures" not in st.session_state:
    st.session_state.admin_login_failures = 0

thread_config = thread_config_for(st.session_state.thread_id)

DISPATCH_VIEW = ":material/forum: Dispatch console"
AUDIT_VIEW = ":material/shield: Security & audit logs"

# Starter questions shown on an empty chat: (icon, label, prompt)
SUGGESTIONS: list[tuple[str, str, str]] = [
    (":material/thermostat:", "Fleet temperature risk", "Which vehicles in the active fleet have the highest temperature risk right now?"),
    (":material/route:", "Corridor conditions", "What are the current corridor conditions near latitude 33.77, longitude -118.19?"),
    (":material/policy:", "Perishables SOP", "What are the temperature rules for fresh perishables?"),
]

# ==========================================
# 5. REUSABLE RENDER HELPERS
# ==========================================
def tool_call_label(name: str | None) -> str:
    """Markdown label for a tool-call accordion (expander labels render badges and code)."""
    return f":orange-badge[:material/bolt: Tool call] `{name}`"


def render_tool_input(name: str | None, args: Any) -> None:
    with st.expander(f"Generated input: {name}", icon=":material/input:", expanded=False):
        st.json(args)


def render_tool_output(name: str | None, content: Any) -> None:
    with st.expander(f"Raw output: {name}", icon=":material/output:", expanded=False):
        st.code(content, language="text")


def render_saved_traces(traces: list[dict[str, Any]]) -> None:
    """One accordion per tool call; opening it reveals its input and raw output accordions."""
    outputs = {str(t.get("id") or t["name"]): t for t in traces if t["type"] == "tool_output"}
    for trace in traces:
        if trace["type"] != "tool_input":
            continue
        with st.expander(tool_call_label(trace["name"]), expanded=False):
            render_tool_input(trace["name"], trace["args"])
            output = outputs.get(str(trace.get("id") or trace["name"]))
            if output is not None:
                render_tool_output(output["name"], output["content"])

# ==========================================
# 6. SIDEBAR NAVIGATION & METADATA
# ==========================================
with st.sidebar:
    st.title(":material/ac_unit: FDE command center")

    app_mode = st.radio("System mode", [DISPATCH_VIEW, AUDIT_VIEW])

    with st.container(border=True):
        st.caption("Session token")
        st.markdown(f"`{st.session_state.thread_id[:8]}...`")
        st.caption("Reasoning architecture")
        st.badge(settings.agent_llm.value, icon=":material/psychology:", color="blue")
        st.caption("Model")
        st.markdown(f"`{settings.active_model}`")

    if st.button("Purge session", icon=":material/delete:", width="stretch"):
        checkpointer.delete_thread(st.session_state.thread_id)  # free the old conversation's memory
        st.session_state.ui_messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ==========================================
# 7. VIEW ROUTING (DISPATCH VS AUDIT)
# ==========================================

if app_mode == DISPATCH_VIEW:
    # ------------------------------------------
    # TAB 1: CHAT UI & AGENT EXECUTION
    # ------------------------------------------
    st.title("Cold-chain incident control")
    st.caption("Real-time decision support for fleet telemetry, corridor conditions and compliance.")

    history = cast(list[dict[str, Any]], st.session_state["ui_messages"])

    for entry in history:
        with st.chat_message(entry["role"]):
            if "traces" in entry:
                render_saved_traces(cast(list[dict[str, Any]], entry["traces"]))
            st.markdown(entry["content"])

    welcome = st.empty()
    suggested: str | None = None
    if not history:
        with welcome.container():
            st.subheader("What do you need to check?")
            st.caption("Ask in plain language, or start with one of these.")
            with st.container(horizontal=True):
                for icon, label, suggestion_prompt in SUGGESTIONS:
                    if st.button(label, icon=icon, key=f"suggest_{label}"):
                        suggested = suggestion_prompt

    typed_input = st.chat_input("Ask about fleet telemetry, corridor conditions or compliance thresholds...")
    user_input = typed_input or suggested

    if user_input:
        welcome.empty()

        history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            final_response = ""
            current_traces: list[dict[str, Any]] = []

            # Status is only a progress row: Streamlit can't nest expanders inside it, so the
            # collapsible tool traces are drawn in a separate container right below it.
            status = st.status("Reasoning about your request...", expanded=False)
            trace_area = st.container()
            tool_slots: dict[str, DeltaGenerator] = {}  # tool_call id -> its group, so output lands next to input
            run_error: str | None = None
            try:
                events = fde_agent.stream(
                    AgentState(messages=[HumanMessage(content=user_input)]),
                    config=thread_config,
                    stream_mode="updates"
                )

                for event in events:
                    for node_name, node_state in event.items():

                        if node_name == "reasoner":
                            latest_msg = node_state["messages"][-1]

                            # A. Intercept Tool Call Requests (Inputs)
                            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                                status.update(label="Preparing tool calls...")
                                for tool_call in latest_msg.tool_calls:
                                    call_id = str(tool_call.get('id') or tool_call['name'])
                                    slot = trace_area.expander(
                                        tool_call_label(tool_call['name']), expanded=False
                                    )
                                    tool_slots[call_id] = slot
                                    with slot:
                                        render_tool_input(tool_call['name'], tool_call['args'])

                                    current_traces.append({
                                        "type": "tool_input",
                                        "id": call_id,
                                        "name": tool_call['name'],
                                        "args": tool_call['args']
                                    })

                                    write_audit_log(
                                        session_id=st.session_state.thread_id,
                                        node_name="reasoner",
                                        tool_name=tool_call['name'],
                                        content=json.dumps(tool_call['args'])
                                    )

                            # B. Intercept Final Generation
                            if latest_msg.content:
                                final_response = latest_msg.content
                                status.update(label="Writing resolution report...")

                                write_audit_log(
                                    session_id=st.session_state.thread_id,
                                    node_name="reasoner_final",
                                    tool_name="LLM Text Synthesis",
                                    content=final_response
                                )

                        elif node_name == "tools":
                            status.update(label="Running tools...")
                            for msg in node_state.get("messages", []):
                                if isinstance(msg, ToolMessage):
                                    output_id = str(msg.tool_call_id or msg.name)
                                    with tool_slots.get(output_id, trace_area):
                                        render_tool_output(msg.name, msg.content)

                                    current_traces.append({
                                        "type": "tool_output",
                                        "id": output_id,
                                        "name": msg.name,
                                        "content": msg.content
                                    })

                                    write_audit_log(
                                        session_id=st.session_state.thread_id,
                                        node_name="tools",
                                        tool_name=msg.name,
                                        content=msg.content
                                    )

                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as e:
                run_error = f"{type(e).__name__}: {e}"
                status.update(label="Agent run failed", state="error", expanded=False)

            if final_response:
                st.markdown(final_response)
                history.append({
                    "role": "assistant",
                    "content": final_response,
                    "traces": current_traces
                })
            elif run_error:
                st.error(f"Agent run failed: {run_error}", icon=":material/error:")
            else:
                st.error("Execution timeout: the engine could not produce a response.", icon=":material/warning:")


elif app_mode == AUDIT_VIEW:
    # ------------------------------------------
    # TAB 2: AUDIT LOG VIEWER (REQUIRES ADMIN CREDENTIALS FROM .ENV OR INPUT)
    # ------------------------------------------
    st.title(":material/shield: Agent audit trail")
    st.caption("Secure inspection of FDE_VIEWS.AgentAuditLog")

    with st.container(border=True):
        st.subheader("Administrator authorization")
        st.caption("Enter the administrative credentials defined in `.env` as `SQL_ADMIN_USER` to query audit logs.")

        with st.form("admin_auth_form", border=False):
            col1, col2 = st.columns(2)
            with col1:
                input_user = st.text_input("Admin username", value=settings.sql_admin_user or "")
            with col2:
                input_pass = st.text_input("Admin password", type="password", value="")

            submit_admin = st.form_submit_button(
                "Authenticate and load logs", icon=":material/lock_open:", type="primary", width="stretch"
            )

    if submit_admin and st.session_state.admin_login_failures >= MAX_ADMIN_LOGIN_FAILURES:
        st.error("Too many failed attempts. Restart the session to try again.", icon=":material/lock:")
    elif submit_admin:
        expected_admin_user = settings.sql_admin_user
        expected_admin_pass = reveal(settings.sql_admin_password)

        credentials_ok = (
            expected_admin_user is not None
            and expected_admin_pass is not None
            and hmac.compare_digest(input_user.encode(), expected_admin_user.encode())
            and hmac.compare_digest(input_pass.encode(), expected_admin_pass.encode())
        )

        if credentials_ok:
            st.session_state.admin_login_failures = 0
            admin_engine: Engine | None = None
            try:
                # Build an isolated admin engine for viewing data
                admin_engine = make_engine(input_user, input_pass)

                with admin_engine.connect() as conn:
                    query = """
                        SELECT TOP 1000 LogID, Timestamp, SessionID, NodeExecuted, ToolName, Content
                        FROM FDE_VIEWS.AgentAuditLog
                        ORDER BY Timestamp DESC
                    """
                    df = pd.read_sql(query, conn)

                st.success("Authenticated as admin.", icon=":material/check_circle:")

                if not df.empty:
                    with st.container(horizontal=True):
                        st.metric("Events", len(df), border=True)
                        st.metric("Sessions", int(df["SessionID"].nunique()), border=True)
                        st.metric("Tools triggered", int(df["ToolName"].nunique()), border=True)

                    cast(Any, st).dataframe(
                        df,
                        column_config={
                            "LogID": st.column_config.NumberColumn("ID", format="%d"),
                            "Timestamp": st.column_config.DatetimeColumn("Execution time", format="DD/MM/YYYY-h:mm a"),
                            "SessionID": "Session token",
                            "NodeExecuted": "Graph node",
                            "ToolName": "Tool triggered",
                            "Content": "Raw payload data"
                        },
                        hide_index=True,
                        width="stretch",
                        height=600
                    )
                else:
                    st.info("No audit logs found yet. Run a query in the dispatch console first.", icon=":material/info:")

            except Exception as e:
                st.error(f"Database query failed: {e}", icon=":material/error:")
            finally:
                if admin_engine is not None:
                    admin_engine.dispose()
        else:
            st.session_state.admin_login_failures += 1
            st.error("Invalid administrator credentials.", icon=":material/error:")
