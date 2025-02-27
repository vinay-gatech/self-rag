from pprint import pprint
from dotenv import load_dotenv
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import chroma_retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.answer_grader import answer_grader, GradeAnswer

load_dotenv()

# def test_foo()->None:
#     assert 1==1

def test_retrieval_grader_yes()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    doc_text    = docs[1].page_content

    res:GradeDocuments  = retrieval_grader.invoke(
        {"question":question, "documents":doc_text}
    )
    assert res.binary_score=="yes"

def test_retrieval_grader_no()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    doc_text    = docs[1].page_content

    res:GradeDocuments  = retrieval_grader.invoke(
        {"question":"how to make pizza", "documents":doc_text}
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

def test_hallucination_grader_yes()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    generation  = generation_chain.invoke({
        "context": docs,
        "question":question
    })
    hallucination_grade:GradeHallucinations = hallucination_grader.invoke({
        "documents":docs,
        "generation":generation
    })

    assert hallucination_grade.binary_score==True

def test_hallucination_grader_no()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    generation  = generation_chain.invoke({
        "context": docs,
        "question":"how to make pizza"
    })
    hallucination_grade:GradeHallucinations = hallucination_grader.invoke({
        "documents":docs,
        "generation":generation
    })

    assert hallucination_grade.binary_score==False

def test_answer_grader_yes()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    generation  = generation_chain.invoke({
        "context": docs,
        "question":question
    })
    answer_grade:GradeAnswer = answer_grader.invoke({
        "question":question,
        "generation":generation
    })

    assert answer_grade.binary_score==True

def test_answer_grader_no()->None:
    question    = "agent memory"
    docs        = chroma_retriever.invoke(question)
    generation  = generation_chain.invoke({
        "context": docs,
        "question":"Mig 25 top speed"
    })
    answer_grade:GradeAnswer = answer_grader.invoke({
        "question":question,
        "generation":generation
    })

    assert answer_grade.binary_score==False