from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools import TavilySearchResults
from graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState)->Dict[str, Any]:
    print("---WEB SEARCH---")
    question        = state["question"]
    documents       = state["documents"]

    tavily_results  = web_search_tool.invoke({"query":question})
    joined_tavily_results   = "\n".join(
        [res["content"] for res in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_results)

    if documents:
        documents.append(web_results)
    else:
        documents   = [web_results]

    return{"documents":documents, "question":question}


if __name__=="__main__":
    print(web_search(state={
        "question":"agent memory",
        "documents": None
    }))
