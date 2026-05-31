from langgraph.graph import StateGraph, START, END
from backend.app.graph.state import AgentState
from backend.app.graph.nodes import llm_node, tool_node, grader_node, rewrite_node
from langgraph.checkpoint.memory import MemorySaver
from backend.app.config import get_settings

MAX_ITERATIONS = 5

def should_continue(state: AgentState) -> str:
    """Determines whether to continue to tools or end the conversation."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # Check max iterations
    if state.get("llm_calls", 0) >= MAX_ITERATIONS:
        return END

    # If the LLM makes a tool call, then route to the "tools" node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
        
    # Otherwise, it has generated the final response
    return END

def grade_documents(state: AgentState) -> str:
    """Determines whether to generate final response or rewrite query."""
    settings = get_settings()
    verdict = state.get("grader_verdict", "yes")
    attempts = state.get("search_attempts", 0)
    
    # If we exceeded max attempts, just force it to the agent to answer as best as possible
    if attempts >= settings.max_search_attempts:
        return "agent"
        
    if verdict == "yes":
        return "agent"
    else:
        return "rewrite_node"

def build_graph():
    """Builds and compiles the Agentic ReAct graph with Self-Correction."""
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node("agent", llm_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_node("grader_node", grader_node)
    graph_builder.add_node("rewrite_node", rewrite_node)
    
    graph_builder.add_edge(START, "agent")
    
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END
        }
    )
    
    # Tool output now goes to grader for relevance check
    graph_builder.add_edge("tool_node", "grader_node")
    
    # Grader decides to either proceed to agent or rewrite the query
    graph_builder.add_conditional_edges(
        "grader_node",
        grade_documents,
        {
            "agent": "agent",
            "rewrite_node": "rewrite_node"
        }
    )
    
    # Rewrite goes back to the agent to try a new search
    graph_builder.add_edge("rewrite_node", "agent")
    
    # Adding MemorySaver for persisting memory across turns in the same thread
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

# Keep a single compiled graph instance with its memory in memory
_compiled_graph = build_graph()

async def astream_agent(session_id: str, message: str):
    """
    Streams updates from the graph.
    Yields dicts containing information about tool calls and messages.
    """
    from langchain_core.messages import HumanMessage
    
    inputs = {
        "messages": [HumanMessage(content=message)], 
        "llm_calls": 0,
        "search_attempts": 0
    }
    config = {"configurable": {"thread_id": session_id}}
    
    # Stream both node updates and individual messages
    async for msg, metadata in _compiled_graph.astream(inputs, config=config, stream_mode="messages"):
        yield msg, metadata

