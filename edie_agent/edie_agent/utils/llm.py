
import os

import dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm(streaming: bool = False, local=False):
    """A helper function to get the LLM instance."""
    dotenv.load_dotenv(dotenv.find_dotenv())

    if local:
        llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
    else:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

    return llm


def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set.")
    return value
