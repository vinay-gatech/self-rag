from typing import Any, Dict
from graph.chains.retrieval_grader import  retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState)->Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question        = state["question"]
    documents       = state["documents"]
    filtered_docs   = []
    use_web_search      = False

    for d in documents:
        score       = retrieval_grader.invoke(
            {"question":question, "documents": documents}
        )

        if score.binary_score=="no":
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            use_web_search  = True
            continue
        else:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)

    return {"question":question, "documents": filtered_docs, "use_web_search": use_web_search}


