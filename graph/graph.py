from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generation_node, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

builder = StateGraph(GraphState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEB_SEARCH, web_search)
builder.add_node(GENERATE, generation_node)

builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)

def websearch_or_generate(state: GraphState)->str:
    if state["use_web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS RELEVANT, INCLUDE WEB SEARCH")
        return WEB_SEARCH
    else:
        print("---DECISION: ALL DOCUMENTS RELEVANT, GENERATE")
        return GENERATE

builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    websearch_or_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    }
)
builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

graph   = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="corrective-rag.png")
