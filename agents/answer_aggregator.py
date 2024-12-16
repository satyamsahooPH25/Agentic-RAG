### Hallucination Grader

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


load_environment_variables()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)


# Data model
class AnswerAggregator(BaseModel):
    """Response aggregator for generating final response from different chunks"""

    answer: str = Field(description="Merged answers of all the intermediate answers")


# LLM with function call
# llm = ChatGroq(
#     model="llama-3.2-90b-vision-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )
structured_llm_grader = llm.with_structured_output(AnswerAggregator)

# Prompt
system = """You are an expert in merging different responses which we get from the different sources.
Your task is to generate a final response, merging list of intermediate responses provided to you.
Make sure that you don't repeat the same statements multiple times. Make it a very detailed and informative answer.
You will get the question for better understanding along with the list of different answers. I will tip you $1000 if you provide a good answer. MAKE DADDY PROUD!!! 

        Question: "{{Input Question}}"
        Answers: "{{List of Answers}}"
"""
aggregation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: \n\n {question} \n\n Answers: \n\n {answers}"),
    ]
)

aggregator = aggregation_prompt | structured_llm_grader

# Example
# question = "What is the difference between a crocodile and an alligator?"
# answers = [
#     "Crocodiles have a pointed snout.",
#     "Alligators have a rounded snout.",
#     "Crocodiles live in saltwater habitats.",
#     "Alligators prefer freshwater."
# ]
# result = aggregator.invoke({"question": question, "answers": answers})
# print(result.answer)
