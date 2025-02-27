from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0)

class GradeAnswer(BaseModel):
    """
    Binary score to check if the answer addresses the question.
    """

    binary_score:bool    = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# LLM bound to HallucinationGrader
structured_llm_grader   = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_grading_prompt   = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question:  \n\n {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader: RunnableSequence   = answer_grading_prompt | structured_llm_grader



