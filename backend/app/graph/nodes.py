import json
from langchain_core.messages import SystemMessage, ToolMessage
from backend.app.graph.state import AgentState
from backend.app.graph.prompts import SYSTEM_PROMPT
from backend.app.services.llm import get_chat_model
from backend.app.tools.search import search_postgres, search_pinecone, fetch_business_details

# Expose our read-only tools to the LLM
tools = [search_postgres, search_pinecone, fetch_business_details]
tools_by_name = {tool.name: tool for tool in tools}

def llm_node(state: AgentState) -> dict:
    """Invokes the LLM to decide the next action or respond to the user."""
    model = get_chat_model()
    model_with_tools = model.bind_tools(tools)
    
    messages = state.get("messages", [])
    
    # Prepend the system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = model_with_tools.invoke(messages)
    
    current_calls = state.get("llm_calls", 0)
    return {"messages": [response], "llm_calls": current_calls + 1}


async def tool_node(state: AgentState) -> dict:
    """Executes the tool calls made by the LLM."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    tool_results = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            if name in tools_by_name:
                tool = tools_by_name[name]
                try:
                    # Invoke tool (sync or async, langchain handles it, but since we are in async node we can await if it's async)
                    observation = await tool.ainvoke(tool_call["args"])
                except Exception as e:
                    observation = json.dumps({"error": f"Tool execution failed: {str(e)}"})
                    
                tool_results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"], name=name))
            else:
                tool_results.append(ToolMessage(content=f"Error: Tool {name} not found.", tool_call_id=tool_call["id"], name=name))
                
    return {"messages": tool_results}
