# Agents - Use the LLM not just for text output, but also for decision making.
# Some applications require not just a predetermined chain of calls to LLMs/other tools, but potentially an unknown chain that depends on the user's input. In these types of chains there is an "agent" which has access to a suite of tools. Depending on the user input, the agent can then decide which, if any, of these tools to call.

# Agent
# The language model that drives decision making.
# More specifically, an agent takes in an input and returns a response corresponding to an action to take along with an action input. There are different types of agents for different use cases.

# Tools
# A 'capability' of an agent. This is an abstraction on top of a function that makes it easy for LLMs (and agents) to interact with it, e.g. Google Search.
# This area shares commonalities with OpenAI plugins.

# Toolkit
# Groups of tools that your agent can select from.
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

llm = OpenAI(temperature=1, openai_api_key=openai_api_key)

toolkit = load_tools(['serpapi'], llm=llm, serpapi_api_key=serpapi_api_key)
agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)

response = agent({"input":"what was the first album of the band that Natalie Bergman is a part of?"})

# print(json.dumps(response["intermediate_steps"], indent=2))
# print(json.dumps(response, indent=2))
