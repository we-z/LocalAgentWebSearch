import requests
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "openai/gpt-oss-20b"

# Tool: fetch DuckDuckGo HTML page
def fetch_webpage(url: str, max_chars=5000) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text[:max_chars]
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"

# Tools definition for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetches the raw HTML content of a web page",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                },
                "required": ["url"],
            },
        },
    }
]

def process_tool_calls(response, messages):
    """Process all tool calls and send results back to the model"""
    tool_calls = response.choices[0].message.get("tool_calls", [])
    if not tool_calls:
        return response.choices[0].message.get("content", "")

    for tool_call in tool_calls:
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        if func_name == "fetch_webpage":
            url = args.get("url")
            result = fetch_webpage(url)
            messages.append({
                "role": "assistant",
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            })

    # Ask the model again after adding tool results
    final_response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return final_response.choices[0].message.content

def chat():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when needed to fetch web content or assist the user."
        }
    ]
    print("Assistant: Hello! I can fetch webpages and summarize them. Just ask anything.")
    print("(Type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Assistant: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Get model response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )

        # Process tool calls if any
        answer = process_tool_calls(response, messages)
        print("\nAssistant:", answer)
        messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    chat()