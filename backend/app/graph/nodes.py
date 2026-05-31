import json
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from backend.app.graph.state import AgentState
from backend.app.graph.prompts import SYSTEM_PROMPT, GRADER_SYSTEM_PROMPT, REWRITE_PROMPT
from backend.app.services.llm import get_chat_model, get_grader_model
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
                    observation = await tool.ainvoke(tool_call["args"])
                except Exception as e:
                    observation = json.dumps({"error": f"Tool execution failed: {str(e)}"})
                    
                tool_results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"], name=name))
            else:
                tool_results.append(ToolMessage(content=f"Error: Tool {name} not found.", tool_call_id=tool_call["id"], name=name))
                
    return {"messages": tool_results}

class Grade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

def grader_node(state: AgentState) -> dict:
    """Determines whether the retrieved documents are relevant to the question."""
    messages = state.get("messages", [])
    
    # Get the original question (the first HumanMessage)
    question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
            
    # Get the latest tool outputs
    docs = ""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            docs += msg.content + "\n"
        else:
            # Once we pass the tool messages, stop
            if docs:
                break
                
    # If no tool messages, default to yes (might be chit-chat)
    if not docs:
        return {"grader_verdict": "yes"}
        
    grader_model = get_grader_model()
    structured_llm_grader = grader_model.with_structured_output(Grade)
    
    prompt = f"{GRADER_SYSTEM_PROMPT}\n\nRetrieved document: \n{docs}\n\nUser question: {question}"
    
    try:
        res = structured_llm_grader.invoke(prompt)
        score = res.binary_score.lower()
    except Exception as e:
        # Fallback if parsing fails
        score = "yes"
        
    # Increment search attempts
    current_attempts = state.get("search_attempts", 0)
    
    return {"search_attempts": current_attempts + 1, "grader_verdict": score}

def rewrite_node(state: AgentState) -> dict:
    """Provides feedback to the agent to rewrite the query."""
    return {"messages": [HumanMessage(content=REWRITE_PROMPT)]}
