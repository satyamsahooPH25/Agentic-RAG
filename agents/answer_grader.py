### Answer Grader

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess if answer addresses question or not."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
# llm = ChatGroq(
#     model="llama-3.2-90b-vision-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer correctly resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes means that the answer is sufficient resolution to the question """
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = answer_prompt | structured_llm_grader

# response=answer_grader.invoke({"question":"What are the key features of the Mars rover, Perseverance", "generation":"erseverance is equipped with a range of advanced scientific instruments, including the Mastcam-Z for capturing high-resolution images, the MOXIE device for generating oxygen from Mars' atmosphere, and the SHERLOC instrument for detecting organic compounds. It also has a miniature helicopter called Ingenuity for aerial exploration, which performed its first flight on Mars in 2021."})
# print(response)
# print(cb.total_tokens)
