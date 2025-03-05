from langchain_openai import ChatOpenAI
from browser_use import Agent
from mem0 import Memory
import asyncio
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Create a Memory instance from the configuration
config = {"vector_store": {"provider": "chroma", "config": {"path": "/tmp/mem0"}}}
memory = Memory.from_config(config_dict=config)

# Define the messages to be added to the vector store
messages = [
    {"role": "user", "content": "Hey, I'm Mark. I love exploring new places, especially in the mountains. I prefer staying in airbnb accommodations and love spicy indian food."},
    {"role": "assistant", "content": "Hey Mark, glad to meet you. I'll remember that!"}
]

# Create memory for the user based on the messages
memory.add(messages, user_id="mark")


# Define the main function
async def main():
    # Create an Agent instance
    from browser_use.browser.context import BrowserContextConfig
    from browser_use.browser.browser import Browser
    from browser_use.browser.context import BrowserContext

    config = BrowserContextConfig(
        browser_window_size={'width': 1920, 'height': 1080},
    )

    browser = Browser()
    context = BrowserContext(browser=browser, config=config)

    agent = Agent(
        task="Planning a trip to San Francisco. Create a personalized itinerary for me for 3 days.",  # Task for the agent
        llm=ChatOpenAI(model="gpt-4o-mini"),  # Language model for the agent
        user_id="mark",  # User ID for the agent to retrieve memories
        browser_context=context,

    )
    # Run the agent and get the result
    result = await agent.run()
    # Print the final result
    print("---")
    print(result.final_result())

# Run the main function
asyncio.run(main())
