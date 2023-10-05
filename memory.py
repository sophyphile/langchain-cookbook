# Memory - Helping LLMs remember information
# Memory is a bit of a lose term. It could be as simple as remembering information you've chatted about in the past or more complicated information retrieval.
# We'll keep it towards the Chat Message use case used for chat bots, but there are many different types of memory to explore in the documentation.

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Chat Message History

from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of france?")

print(history.messages)

ai_response = chat(history.messages)
print(ai_response)

history.add_ai_message(ai_response.content)
print(history.messages)

# LangChain even helps us save this chat history for later use.