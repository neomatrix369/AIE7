from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

"""
Add your own tool here, and then use it through Cursor!
"""
@mcp.tool()
def country_info(country_name: str) -> dict:
    """Get information about a country"""
    import requests
    # search for articles
    search_url = f"https://restcountries.com/v3.1/name/{country_name}"
    response = requests.get(search_url)
    data = response.json()[0]
    return {
        "search_url": search_url,
        "name": data["name"]["common"],
        "capital": data.get("capital", ["N/A"])[0],
        "population": data["population"],
        "currency": list(data.get("currencies", {}).keys())[0] if data.get("currencies") else "N/A"
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")