

from src.agent.base_agent import Agent
from dotenv import load_dotenv

# load the GROQ API KEY from a .env file
load_dotenv("../.env")

if __name__ == '__main__':

    agent = Agent(agent_id=0, action_space=[])

    agent.step("Hey, how are you doing?")


