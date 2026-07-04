
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import logging
import warnings

# Sab warnings band karo
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)

load_dotenv()
import asyncio

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()
    model = ChatGroq(model="qwen/qwen3-32b")
    agent = create_react_agent(model, tools)

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12? Give answer in plain text only, no LaTeX or markdown formatting."}]}
    )
    print("Math response:", math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's the weather in california? Give a short plain text answer only."}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)

asyncio.run(main())
