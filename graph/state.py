from typing import TypedDict, List

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        use_web_search: whether to add search
        documents: list of documents
    """

    question:       str
    generation:     str
    use_web_search: bool
    documents:      List[str]