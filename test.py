import os
import logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(asctime)s - %(pathname)s - %(lineno)s - %(message)s')

from langchain_openai import ChatOpenAI
from browser_use import Agent

from mem0 import Memory

import asyncio
from dotenv import load_dotenv
load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "false"
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "/tmp/mem0",
        }
    }
}


memory = Memory.from_config(config_dict=config)
memory.add("Remember my name is Deshraj and I am avid traveler. I like to stay in airbnb and travel to new places preferably in mountains", user_id="deshraj")


async def main():
    agent = Agent(
        task="Planning to go somewhere near San Francisco. Create an itinerary and find a place to stay.",
        llm=ChatOpenAI(model="gpt-4o-mini"),
        user_id="deshraj",
    )
    result = await agent.run()
    print(result.final_result())
    # print(result)

asyncio.run(main())
