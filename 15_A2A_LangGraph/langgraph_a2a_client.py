"""
LangGraph Agent that uses A2A Protocol to interact with the helpfulness agent.

This creates a simple agent that can make API calls to the A2A Agent Node
through the A2A protocol, demonstrating agent-to-agent communication.
"""

import asyncio
import logging
from typing import Dict, Any, List, Annotated
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

from dotenv import load_dotenv

# Load .env file
load_dotenv()

class A2AAgentState(TypedDict):
    """State schema for our A2A LangGraph agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    context_id: str | None
    task_id: str | None


@tool
async def call_a2a_agent(query: str, context_id: str = "", task_id: str = "") -> str:
    """
    Call the A2A helpfulness agent with a query.
    
    Args:
        query: The question or request to send to the A2A agent
        context_id: Optional context ID for multi-turn conversations
        task_id: Optional task ID for multi-turn conversations
        
    Returns:
        The response from the A2A agent
    """
    base_url = 'http://localhost:10000'
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            # Initialize resolver and get agent card
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            
            # Initialize A2A client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Prepare message payload
            message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': query}],
                    'message_id': uuid4().hex,
                }
            }
            
            # Add context/task IDs if provided for multi-turn conversation
            if context_id and context_id.strip():
                message_payload['message']['context_id'] = context_id
            if task_id and task_id.strip():
                message_payload['message']['task_id'] = task_id
            
            # Send request
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**message_payload)
            )
            
            response = await client.send_message(request)
            
            # Extract response text
            if hasattr(response.root.result, 'response') \
                and response.root.result.response \
                and hasattr(response.root.result.response, 'parts'):
                parts = response.root.result.response.parts
                if parts and len(parts) > 0:
                    return parts[0].get('text', 'No text in response')
            
            return f"A2A Agent Response: {response.model_dump(mode='json', exclude_none=True)}"
            
    except Exception as e:
        return f"Error calling A2A agent: {str(e)}"


def create_a2a_langgraph_agent():
    """Create a LangGraph agent that uses A2A protocol for communication."""
    
    # Initialize the LLM for our agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    llm_with_tools = llm.bind_tools([call_a2a_agent])
    
    def agent_node(state: A2AAgentState) -> Dict[str, Any]:
        """Main agent node that decides whether to use A2A or respond directly."""
        messages = state["messages"]
        
        # Call the LLM with tools
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    async def a2a_tool_node(state: A2AAgentState) -> Dict[str, Any]:
        """Execute A2A tool calls."""
        messages = state["messages"]
        last_message = messages[-1]
        
        results = []
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "call_a2a_agent":
                    # Extract arguments
                    args = tool_call["args"]
                    query = args.get("query", "")
                    context_id = state.get("context_id") or ""
                    task_id = state.get("task_id") or ""
                    
                    # Call A2A agent using proper tool invocation
                    result = await call_a2a_agent.ainvoke({
                        "query": query, 
                        "context_id": context_id, 
                        "task_id": task_id
                    })
                    
                    # Create tool message
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                    results.append(tool_message)
        
        return {"messages": results}
    
    def should_continue(state: A2AAgentState) -> str:
        """Determine whether to call tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return END
    
    # Build the graph
    graph = StateGraph(A2AAgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", a2a_tool_node)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Add edge from tools back to agent
    graph.add_edge("tools", "agent")
    
    return graph.compile()


async def test_a2a_langgraph_agent():
    """Test the A2A LangGraph agent."""
    print("🤖 Testing A2A LangGraph Agent")
    print("=" * 50)
    
    # Create the agent
    agent = create_a2a_langgraph_agent()
    
    # Test queries
    test_queries = [
        "What are the latest developments in artificial intelligence?",
        "Can you help me understand transformer architectures?",
        "What are the benefits of using A2A protocols for agent communication?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 40)
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "context_id": None,
            "task_id": None
        }
        
        try:
            # Run the agent
            result = await agent.ainvoke(initial_state)
            
            # Print results
            final_message = result["messages"][-1]
            print(f"Agent Response: {final_message.content}")
            print(f"Total messages: {len(result['messages'])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ A2A LangGraph Agent testing complete!")


def create_simple_a2a_agent():
    """Create a simpler version for basic testing."""
    
    async def simple_agent(query: str) -> str:
        """Simple agent that directly calls A2A."""
        print(f"🤖 Simple A2A Agent received: {query}")
        
        # Invoke the tool properly using .ainvoke
        result = await call_a2a_agent.ainvoke({"query": query})
        return result
    
    return simple_agent


async def test_simple_a2a_agent():
    """Test the simple A2A agent."""
    print("🤖 Testing Simple A2A Agent")
    print("=" * 30)
    
    agent = create_simple_a2a_agent()
    
    test_query = "What are the key components of student loan repayment?"
    print(f"Query: {test_query}")
    
    try:
        response = await agent(test_query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Main function to run tests."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting A2A LangGraph Agent Tests")
    print("=" * 60)
    
    # Test 1: Simple agent
    await test_simple_a2a_agent()
    
    print("\n" + "=" * 60)
    
    # Test 2: Full LangGraph agent
    await test_a2a_langgraph_agent()


if __name__ == "__main__":
    asyncio.run(main())