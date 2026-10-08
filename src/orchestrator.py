import io
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Annotated, Any, Literal, Protocol, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models import LanguageModelInput
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Emoji log lines crash on Windows consoles/pipes using cp1252; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 1. SETUP & PATH RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[0]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AgentLLM, settings
from src.agent_tools import query_telemetry_db, fetch_corridor_conditions, search_compliance_sop

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class FdeAgent(Protocol):
    """The slice of the compiled LangGraph API this project uses."""
    def stream(
        self, input: AgentState, config: RunnableConfig, *, stream_mode: Literal["updates"]
    ) -> Iterator[dict[str, Any]]: ...

# ==========================================
# 2. FACTORY INITIALIZATION: AGENT REASONER LLM
# ==========================================

AGENT_LLM_SETTING: AgentLLM = settings.agent_llm
llm: BaseChatModel

if AGENT_LLM_SETTING is AgentLLM.OPENAI:
    print(f"🤖 Brain Mode: Utilizing Cloud OpenAI Reasoner ({settings.openai_model})...")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)

elif AGENT_LLM_SETTING is AgentLLM.DEEPSEEK:
    print(f"🐳 Brain Mode: Utilizing DeepSeek Cloud Reasoner ({settings.deepseek_model})...")
    from langchain_openai import ChatOpenAI

    deepseek_api_key = settings.deepseek_api_key
    if deepseek_api_key is None:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in .env but Agent_llm=DEEPSEEK.")

    # Fully updated to match 2026 DeepSeek API parameters and endpoint contracts
    llm = ChatOpenAI(
        model=settings.deepseek_model,                           # deepseek-v4-flash, deepseek-v4-pro
        temperature=0,
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",     # Fixed connection string url endpoint
        model_kwargs={"max_tokens": 2048},       # Passed through verbatim; DeepSeek expects "max_tokens"
        # extra_body={
        #     "thinking": {"type": "enabled"},              # Activates DeepSeek Deep-Thinking mode
        #     "reasoning_effort": "high"                     # Drives maximal reasoning depth for logic maps
        # }
    )

else:  # FALLBACK / DEFAULT RUNNER MODE
    print(f"🤗 Brain Mode: Local Fallback Activated. Binding Local Ollama ({settings.ollama_model})...")
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=settings.ollama_model, temperature=0, num_predict=1024, keep_alive="30m")  # keep_alive: skip the model reload between questions

fde_tools = [query_telemetry_db, fetch_corridor_conditions, search_compliance_sop]
llm_with_tools: Runnable[LanguageModelInput, AIMessage] = cast(Any, llm).bind_tools(fde_tools)

# ==========================================
# 3. GRAPH ARCHITECTURE ASSEMBLY
# ==========================================
def load_system_prompt() -> SystemMessage:
    """Load the business system prompt; fall back to a basic one if missing."""
    prompt_path = project_root / "src" / "prompts" / "system_prompt.txt"
    try:
        return SystemMessage(content=prompt_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Warning: Could not find {prompt_path}; using basic fallback prompt.")
        return SystemMessage(content="You are a helpful AI assistant.")

SYSTEM_PROMPT = load_system_prompt()

# Bounds so a long chat or a looping model can't grow the prompt (slower calls,
# context overflow) or run up an unbounded number of slow LLM steps.
MAX_HISTORY_MESSAGES = 20
RECURSION_LIMIT = 12

# Checkpointed chat state lives in memory; the UI deletes a thread when it is purged.
checkpointer = MemorySaver()


def thread_config_for(thread_id: str) -> RunnableConfig:
    """Per-conversation run config with the step cap applied."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

def reasoning_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    # System prompt is injected per call (not stored in state) so every client
    # (CLI, UI) gets it without a wasted initial LLM round-trip.
    recent = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,  # count messages, not tokens
        max_tokens=MAX_HISTORY_MESSAGES,
        start_on="human",   # never begin the window on an orphaned tool result
        allow_partial=False,
    )
    history: list[BaseMessage] = [SYSTEM_PROMPT, *(recent or state["messages"])]
    response = llm_with_tools.invoke(history)
    return {"messages": [response]}

print("⚙️ Compiling LangGraph FDE Orchestrator...")
graph_builder = cast(Any, StateGraph(AgentState))
graph_builder.add_node("reasoner", reasoning_node)
graph_builder.add_node("tools", ToolNode(fde_tools))

graph_builder.add_edge(START, "reasoner")
graph_builder.add_conditional_edges("reasoner", tools_condition)
graph_builder.add_edge("tools", "reasoner")

fde_agent: FdeAgent = graph_builder.compile(checkpointer=checkpointer)

# ==========================================
# 4. CHAT LOOP TESTING PANEL
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("🚀 FDE Supply Chain Orchestrator State Machine Online")
    print(f"   Configured Execution: [LLM: {AGENT_LLM_SETTING}] -> [Embeddings: {settings.embeddings_model}]")
    print("="*55 + "\n")
    
    thread_config = thread_config_for("production_test_1")

    while True:
        try:
            user_input = input("\nDispatcher > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            events = fde_agent.stream(
                AgentState(messages=[HumanMessage(content=user_input)]),
                config=thread_config,
                stream_mode="updates",
            )
            for event in events:
                for node_name, node_state in event.items():
                    if node_name == "tools":
                        print("   [System] 🔄 Retrieving external data elements via ToolNode...")
                    elif node_name == "reasoner":
                        latest_msg = cast(dict[str, list[BaseMessage]], node_state)["messages"][-1]
                        if latest_msg.content:
                            print(f"\n🤖 FDE Agent:\n{latest_msg.content}")
        except Exception as e:
            print(f"\n⚠️ Agent run failed: {type(e).__name__}: {e}")