### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


# Data model
class RelevancyRouting(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["answerable", "not_answerable"] = Field(
        ...,
        description="Given a user question choose to route it based on whether it is answerable or not with given context retrieved.",
    )


# LLM with function call
# llm = ChatGroq(
#     model="llama-3.2-90b-text-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)

structured_llm_router = llm.with_structured_output(RelevancyRouting)

# Prompt
system = """You are an expert at routing a user question based on whether the question can be answered with the given context or not.
If the retrieved documents have information sufficient to answer the question, then the question is answerable.
Otherwise, the question is not answerable."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the question: {question} \n\n Here is the context: {context}",
        ),
    ]
)

relevency_router = route_prompt | structured_llm_router
