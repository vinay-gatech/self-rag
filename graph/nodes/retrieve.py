from typing import Any, Dict
from dotenv import load_dotenv
from graph.state import GraphState
from ingestion import chroma_retriever

load_dotenv()

def retrieve(state:GraphState)->Dict[str,Any]:
    print("---RETRIEVE---")
    question    = state["question"]
    retrieved_docs  = chroma_retriever.invoke(question)
    return {
        "documents":retrieved_docs,
        "question":question
    }
graph_state = GraphState()
graph_state["question"] = "What is an LLM"

print(retrieve(graph_state))