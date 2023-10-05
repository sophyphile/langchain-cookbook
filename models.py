# Models - The interface to the AI brains
# Important because there are various models and model types
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Language Model
# A model that does text in -> text out
# from langchain.llms import OpenAI

# llm = OpenAI(model_name="text-ada-001", openai_api_key=openai_api_key)
# print(llm("What day comes after Friday?"))

# Chat Model
# A model that takes a series of messages and returns a message output
# from langchain.chat_models import ChatOpenAI
# from langchain.schema import HumanMessage, SystemMessage, AIMessage

# chat = ChatOpenAI(temperature=1, openai_api_key=openai_api_key)

# print(chat(
#     [
#         SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says"),
#         HumanMessage(content="I would like to go to New York, how do I do this?")
#     ]
# ))

# Text Embedding Model
# Change your text into a vector (a series of numbers that hold the semantic 'meaning' of your text). Mainly used when comparing two pieces of text together.

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

text = "Hi! It's time for the beach."

text_embedding = embeddings.embed_query(text)
print (f"Your embedding is length {len(text_embedding)}")
print (f"Here's a sample: {text_embedding[:5]}...")
