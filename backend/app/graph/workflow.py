from langgraph.graph import StateGraph, START, END
from backend.app.graph.state import AgentState
from backend.app.graph.nodes import llm_node, tool_node
from langgraph.checkpoint.memory import MemorySaver

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

def build_graph():
    """Builds and compiles the Agentic ReAct graph."""
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node("agent", llm_node)
    graph_builder.add_node("tool_node", tool_node)
    
    graph_builder.add_edge(START, "agent")
    
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END
        }
    )
    
    graph_builder.add_edge("tool_node", "agent")
    
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
    
    inputs = {"messages": [HumanMessage(content=message)], "llm_calls": 0}
    config = {"configurable": {"thread_id": session_id}}
    
    # Stream both node updates and individual messages
    async for msg, metadata in _compiled_graph.astream(inputs, config=config, stream_mode="messages"):
        yield msg, metadata

