from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generation_node, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.answer_grader import answer_grader, GradeAnswer

load_dotenv()


def websearch_or_generate(state: GraphState)->str:
    if state["use_web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS RELEVANT, INCLUDE WEB SEARCH")
        return WEB_SEARCH
    else:
        print("---DECISION: ALL DOCUMENTS RELEVANT, GENERATE")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState)->str:
    print("---CHECK HALLUCINATIONS---")
    question    = state["question"]
    documents   = state["documents"]
    generation  = state["generation"]

    print("---GRADE: HALLUCINATION IN GENERATION---")
    hallucination_grade = hallucination_grader.invoke({
        "documents" : documents,
        "generation": generation
    })

    if hallucination_grade.binary_score==True:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE: GENERATION ADDRESSES QUESTION---")
        answer_grade    = answer_grader.invoke({
            "question" : question,
            "generation": generation
        })
        if answer_grade.binary_score==True:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"


    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-GENERATE---")
        return "not supported"


builder = StateGraph(GraphState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEB_SEARCH, web_search)
builder.add_node(GENERATE, generation_node)

builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)

builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    websearch_or_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    }
)
builder.add_edge(WEB_SEARCH, GENERATE)
# builder.add_edge(GENERATE, END)
builder.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map=
    {
        "useful":END,
        "not useful":WEB_SEARCH,
        "not supported":GENERATE
    }
)

graph   = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="corrective-rag.png")
