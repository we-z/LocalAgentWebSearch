from openai import OpenAI
from ddgs import DDGS
import json

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL = "openai/gpt-oss-20b"

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the internet for up-to-date information.",
        "parameters": {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        },
    },
}

def search_web(query):
    results = DDGS().text(query, max_results=5)
    if results:
        return json.dumps(results, indent=2)
    return "No results found."

def process_stream(stream, add_assistant_label=True):
    """Handle streaming responses from the API"""
    collected_text = ""
    tool_calls = []
    first_chunk = True

    for chunk in stream:
        delta = chunk.choices[0].delta

        # Handle regular text output
        if delta.content:
            if first_chunk:
                print()
                if add_assistant_label:
                    print("Assistant:", end=" ", flush=True)
                first_chunk = False
            print(delta.content, end="", flush=True)
            collected_text += delta.content

        # Handle tool calls
        elif delta.tool_calls:
            for tc in delta.tool_calls:
                if len(tool_calls) <= tc.index:
                    tool_calls.append({
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}
                    })
                tool_calls[tc.index] = {
                    "id": (tool_calls[tc.index]["id"] + (tc.id or "")),
                    "type": "function",
                    "function": {
                        "name": (tool_calls[tc.index]["function"]["name"] + (tc.function.name or "")),
                        "arguments": (tool_calls[tc.index]["function"]["arguments"] + (tc.function.arguments or ""))
                    }
                }
    return collected_text, tool_calls

def chat_loop():
    messages = []
    print("Assistant: Hi! I am an AI agent empowered with the ability to search the web (Type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        # Get initial response
        response_text, tool_calls = process_stream(
            client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[SEARCH_TOOL],
                stream=True,
                temperature=0.2
            )
        )

        if not tool_calls:
            print()

        text_in_first_response = len(response_text) > 0
        if text_in_first_response:
            messages.append({"role": "assistant", "content": response_text})

        # Handle tool calls if any
        if tool_calls:
            tool_name = tool_calls[0]["function"]["name"]
            print()
            if not text_in_first_response:
                print("Assistant:", end=" ", flush=True)
            print(f"**Calling Tool: {tool_name}**")
            messages.append({"role": "assistant", "tool_calls": tool_calls})

            # Execute tool calls
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == "search_web":
                    arguments = json.loads(tool_call["function"]["arguments"])
                    query = arguments.get("query")
                    result = search_web(query)
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call["id"]
                    })

            # Get final response after tool execution
            final_response, _ = process_stream(
                client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    stream=True
                ),
                add_assistant_label=False
            )

            if final_response:
                print()
                messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    chat_loop()
