import json
import requests
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "openai/gpt-oss-20b"


def web_search_duckduckgo(query: str, max_results: int = 5) -> dict:
    """
    Perform a DuckDuckGo search and return the top results.
    
    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return.

    Returns:
        dict: Search results including titles and URLs.
    """
    try:
        search_url = "https://duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/116.0.0.0 Safari/537.36"
        }
        params = {"q": query}

        response = requests.post(search_url, headers=headers, data=params, timeout=10)
        response.raise_for_status()

        # Parse links from HTML using simple string operations
        results = []
        html = response.text.split('<a rel="nofollow" class="result__a" href="')
        for entry in html[1:max_results+1]:
            url_part = entry.split('"')[0]
            title_part = entry.split('">')[1].split('</a>')[0]
            results.append({"title": title_part, "url": url_part})

        return {"status": "success", "query": query, "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search_duckduckgo",
            "description": "Search the web using DuckDuckGo and return the top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def process_tool_calls(response, messages):
    """Process a tool call from the assistant and return the final response"""
    tool_calls = response.choices[0].message.tool_calls
    messages.append({
        "role": "assistant",
        "tool_calls": [{"id": tc.id, "type": tc.type, "function": tc.function} for tc in tool_calls]
    })

    for tool_call in tool_calls:
        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments.strip() else {}
        if tool_call.function.name == "web_search_duckduckgo":
            result = web_search_duckduckgo(**args)
        else:
            result = {"status": "error", "message": f"Unknown tool {tool_call.function.name}"}

        tool_result_message = {
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id,
        }
        messages.append(tool_result_message)

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return final_response


def chat():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can perform web searches using DuckDuckGo for up-to-date information."
        }
    ]

    print("Assistant: Hello! I can help you search the web using DuckDuckGo for up-to-date info.")
    print("(Type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Assistant: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )

            if response.choices[0].message.tool_calls:
                final_response = process_tool_calls(response, messages)
                print("\nAssistant:", final_response.choices[0].message.content)
                messages.append({"role": "assistant", "content": final_response.choices[0].message.content})
            else:
                print("\nAssistant:", response.choices[0].message.content)
                messages.append({"role": "assistant", "content": response.choices[0].message.content})

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            break


if __name__ == "__main__":
    chat()