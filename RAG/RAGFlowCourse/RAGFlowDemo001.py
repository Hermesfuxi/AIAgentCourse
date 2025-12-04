import os

from dotenv import load_dotenv, find_dotenv
from ragflow_sdk import RAGFlow

load_dotenv(find_dotenv())

RAGFLOW_BASE_URL = os.environ["RAGFLOW_BASE_URL"]
RAGFLOW_API_URL = os.environ["RAGFLOW_API_URL"]
RAGFLOW_API_KEY = os.environ["RAGFLOW_API_KEY"]

rag_object = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)

chats = rag_object.list_chats()
print(chats[1].llm) # {'frequency_penalty': 0.5, 'model_name': 'Qwen/Qwen2.5-7B-Instruct___OpenAI-API@OpenAI-API-Compatible', 'presence_penalty': 0.5, 'temperature': 0.2, 'top_p': 0.75}

chats = rag_object.list_chats(id='86936f54c47f11f097350242ac150006')
chat = chats[0]
chat.update(update_message=dict(llm={
    "model_name": "gpt-3.5-turbo",
    },
    model_params={"temperature": 0.5})
)
# rag_object.create_chat(name="Demo Chat", avatar="https://avatars.githubusercontent.com/u/13991007?s=200&v=4", dataset_ids=[1], llm="gpt-3.5-turbo", model_params={"temperature": 0.5})
# print(chat.create_session(name="Demo Session"))
