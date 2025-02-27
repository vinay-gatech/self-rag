from pprint import pprint
from dotenv import load_dotenv
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import chroma_retriever
from graph.chains.generation import generation_chain
load_dotenv()

# def test_foo()->None:
#     assert 1==1

def test_retrieval_grader_yes()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    doc_text    = docs[1].page_content

    res:GradeDocuments  = retrieval_grader.invoke(
        {"question":question, "document":doc_text}
    )
    assert res.binary_score=="yes"

def test_retrieval_grader_no()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    doc_text    = docs[1].page_content

    res:GradeDocuments  = retrieval_grader.invoke(
        {"question":"how to make pizza", "document":doc_text}
    )
    assert res.binary_score=="no"

def test_generation_chain()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    generation  = generation_chain.invoke({
        "context": docs,
        "question":question
    })
    pprint(generation)
