import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # LangGraph automatically appends messages to this list
    messages: Annotated[list[AnyMessage], operator.add]
    # Tracks the number of LLM tool-calling loops to prevent infinite loops
    llm_calls: int
