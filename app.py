import streamlit as st
import asyncio
from backend.app.graph.workflow import astream_agent
from dotenv import load_dotenv
import json

load_dotenv()

st.set_page_config(page_title="Karobar Online - Enterprise Agentic Chatbot", page_icon="🤖")

st.title("Karobar Online Agentic Chatbot")
st.markdown("This is the pure Agentic ReAct version of the chatbot. It autonomously decides when to search Postgres, Pinecone, or fetch detailed business records.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I can help you find businesses, shops, and services across Pakistan. What are you looking for today?"}
    ]

# Render past history natively in streamlit
for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if "trace" in msg and msg["trace"]:
                for t in msg["trace"]:
                    with st.status(t["label"], state="complete"):
                        st.json(t["data"])
            st.write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

async def process_stream(prompt):
    session_id = "default_session" # Memory thread ID
    
    from langchain_core.messages import AIMessageChunk, ToolMessage
    
    with st.chat_message("assistant"):
        status = st.status("Agent is thinking...", state="running")
        trace_data = []
        
        response_placeholder = st.empty()
        full_response = ""
        seen_tool_calls = set()
        
        async for chunk, metadata in astream_agent(session_id, prompt):
            # Handle AI Message Chunks (can contain tool calls or text)
            if isinstance(chunk, AIMessageChunk):
                # Handle tool call streaming
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    for tc_chunk in chunk.tool_call_chunks:
                        tc_id = tc_chunk.get("id")
                        if tc_id and tc_id not in seen_tool_calls:
                            seen_tool_calls.add(tc_id)
                            name = tc_chunk.get("name") or "tool"
                            label = f"🛠️ Calling Tool: `{name}`"
                            status.write(f"**{label}**")
                            trace_data.append({"label": label, "data": "Arguments streaming..."})
                
                # Handle standard text streaming (the final response)
                if chunk.content and isinstance(chunk.content, str):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
                    status.update(label="Response generated", state="complete")
            
            # Handle Tool Results
            elif isinstance(chunk, ToolMessage):
                label = f"✅ Tool Result: `{chunk.name}`"
                status.write(f"**{label}**")
                try:
                    res = json.loads(chunk.content)
                    status.json(res)
                    trace_data.append({"label": label, "data": res})
                except:
                    status.write(chunk.content)
                    trace_data.append({"label": label, "data": chunk.content})
        
        # Remove the typing cursor at the end
        if full_response:
            response_placeholder.markdown(full_response)
        else:
            response_placeholder.markdown("*(No text response generated)*")
        
        # Save to session state
        st.session_state["messages"].append({
            "role": "assistant", 
            "content": full_response,
            "trace": trace_data
        })

if prompt := st.chat_input("Ask about a business..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    asyncio.run(process_stream(prompt))

