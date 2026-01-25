import os
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from tools import tools_list

# Load environment variables (.env)
load_dotenv()

# --- Configuration ---
# DB_PATH is not needed for MemorySaver, but kept for reference
DB_PATH = "firewatch_chat_history.db"

# --- State Definition ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Model Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-09-2025",
    temperature=0.4, 
    max_retries=2,
)

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools_list)

# --- Nodes ---
def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# --- Graph Construction ---
def build_firewatch_graph():
    graph_builder = StateGraph(State)
    
    # Add Nodes
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", ToolNode(tools_list))
    
    # Add Edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition, 
    )
    graph_builder.add_edge("tools", "chatbot")
    
    return graph_builder

# --- System Prompt ---
SYSTEM_PROMPT = SystemMessage(content="""
You are FireWatch AI, a tactical operations assistant.
Your Goal: Provide rapid decision support to First Responders.

Context & Data:
- You have tools to fetch LIVE fire data. USE THEM when asked about locations.
- Confidence: HIGH (>80%) = Active Emergency. MEDIUM = Verify. LOW = Monitor.

Guidelines:
- Be Concise. Bullet points.
- No Hallucinations: If the tool returns no data, say "No verified data."
- Action-Oriented: Suggest deployments (e.g., "Deploy drone to Sector 4").
""")

# --- Helper Class for Compatibility ---
# This mimics the database connection behavior so index.py doesn't crash
class RAMConnection:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

# --- Initializer used by index.py ---
async def get_graph_app():
    # We use MemorySaver (RAM) to avoid complex SQLite async locking issues
    checkpointer = MemorySaver()
    
    # We create a dummy connection object to satisfy index.py's lifespan manager
    conn = RAMConnection()
    
    graph = build_firewatch_graph()
    app = graph.compile(checkpointer=checkpointer)
    return app, conn


if __name__ == "__main__":
    async def main():
        pass