from typing import Tuple, List
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore
from api.config import CHAT_MODEL, SYSTEM_PROMPT, TOP_K, EMBEDDING_MODEL
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "sk-lO1-SW0nLvZiLqO7eI5-Jg"
os.environ["OPENAI_BASE_URL"] = "https://api.llmod.ai/v1"
os.environ["PINECONE_API_KEY"] = "pcsk_SmkCz_4Q2cedbHACYGAxf4ULUVQURyAVqw9D5T8oFSDneqP1ximmTtBtHeSS2te5NAWQx"


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant TED Talk transcript chunks to support answering a query.

    Returns:
        - Serialized context (string) for the LLM
        - Raw documents (artifacts) for inspection / debugging
    """
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="ted",
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL)
    )

    retrieved_docs = vectorstore.similarity_search_with_score(
        query,
        k=TOP_K,
    )

    context = []
    for doc in retrieved_docs:
        context.append({
            "talk_id": doc[0].metadata.get("talk_id"),
            "title": doc[0].metadata.get("title"),
            "chunk": doc[0].page_content,
            "score": doc[1]
        })

    serialized = "\n\n".join(
        (f"Source: {doc[0].metadata}\nContent: {doc[0].page_content}")
        for doc in retrieved_docs
    )
    return serialized, context


def build_agent():
    """
    Create a constrained LangChain agent for TED Talk RAG.
    """
    model = init_chat_model(
        model_provider="openai",
        model=CHAT_MODEL
    )

    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def generate_answer(query):
    agent = build_agent()

    for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
    ):
        msg = event["messages"][-1]

        if msg.type == "tool" and msg.name == "retrieve_context":
            retrieved_docs = msg.artifact

        # Capture final LLM answer
        if msg.type == "ai":
            final_llm_message = msg

    return {
        "response": final_llm_message.content,
        "context": [{
            "talk_id": doc["talk_id"],
            "title": doc["title"],
            "chunk": doc["chunk"],
            "score": doc["score"]
        } for doc in retrieved_docs],
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": query
        }
    }
