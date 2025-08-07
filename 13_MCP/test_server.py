from fastmcp import Client

from pprint import pprint

async def main():
    # Connect via stdio to a local script
    async with Client("server.py") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        print()
        result = await client.call_tool("web_search", {"query": "What is the capital of France?"})
        print(f"Result: {result}")
        print()
        result = await client.call_tool("country_info", {"country_name": "UK"})
        print("Result:")
        pprint(result)
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())