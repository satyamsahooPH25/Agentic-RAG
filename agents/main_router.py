### Router

import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
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
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
# system = """You are an expert at routing a user question to a vectorstore or web search.
# The vectorstore contains 10Q reports of the firm Apple.
# Use the vectorstore for questions on these topics. Otherwise, use web-search."""

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains a 10K report of the firm Alphabet Inc, which is the parent company of Google.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# resp=question_router.invoke("Give me details about apple company?")
# print(resp)
