"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from .models import get_openai_model
from .rag import ProductionRAGChain


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def tool_call_or_helpful(state):
    """Determine next step: tool call, helpfulness check, or end."""
    last_message = state["messages"][-1]

    # If the last message has tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"

    # Check if we've reached maximum conversation length
    if len(state["messages"]) > 10:
        return END

    # Only check helpfulness if we have at least a query and response
    if len(state["messages"]) < 2:
        return "continue"

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    prompt_template = """\
    Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

    Initial Query:
    {initial_query}

    Final Response:
    {final_response}"""

    helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
    helpfulness_check_model = ChatOpenAI(model="gpt-4o-mini")
    helpfulness_chain = helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()

    try:
        helpfulness_response = helpfulness_chain.invoke({
            "initial_query": initial_query.content, 
            "final_response": final_response.content
        })
        
        return "end" if "Y" in helpfulness_response else "continue"
    except Exception as e:
        print(f"Error in helpfulness check: {e}")
        return "continue"


def create_langgraph_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a LangGraph agent following the helpfulness agent pattern.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages - agent node."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Build graph following helpfulness agent pattern
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes: agent (model) and action (tools)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    
    # Set entry point to agent
    graph.set_entry_point("agent")
    
    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        tool_call_or_helpful,
        {
            "continue" : "agent",
            "action" : "action",
            "end" : END
        }
    )
    
    # Add edge from action back to agent (enables cycles)
    graph.add_edge("action", "agent")
    
    return graph.compile()
