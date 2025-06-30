import os
from langchain.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> str:
    """Search the web for information using the Tavily search API."""
    # Initialize the Tavily client with API key from environment variables
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    
    # Perform the search
    response = client.search(query=query)
    
    results = []
    for i, result in enumerate(response.get("results", []), 1):
        results.append(
            f"{i}. **Title:** {result.get('title')}\n"
            f"   **URL:** {result.get('url')}\n"
            f"   **Content:** {result.get('content')}\n"
        )
    formatted_response = f"Search results for '{query}':\n\n" + "\n".join(results)
    return formatted_response


if __name__ == "__main__":
    tavily_tool = web_search("네이버에서 강아지 옷 찾아줘.")

