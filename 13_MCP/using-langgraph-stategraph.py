from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio

load_dotenv()

import os

from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4.1")

client = MultiServerMCPClient(
    {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": [os.getcwd() + "/math_server.py"],
                "transport": "stdio",
            },

            "mcp-server": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": [os.getcwd() + "/server.py"],
                "transport": "stdio",
            }
        }
)

async def main():
    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    
    math_response = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
    roll_a_dice_response = await graph.ainvoke({"messages": "roll a dice, twice"})
    
    print()
    print("Math response:", math_response)
    print()
    print("Roll-a-dice response:", roll_a_dice_response)
    print()

if __name__ == "__main__":
    asyncio.run(main())