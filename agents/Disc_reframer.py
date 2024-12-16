### Question Re-writer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)


# Prompt
system = """You are an intelligent assistant specializing in question refinement. Your task is to adjust and improve questions based on the provided suggestions.

Instructions:
Receive an input question and a corresponding suggestion for improvement.
Modify atmost two questions the question to align with the suggestion while maintaining clarity, grammatical correctness, and context.
Ensure the revised question is concise and adheres to the original intent unless explicitly instructed otherwise.

"""
Disc_reframe_prmpt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human", "{user_input}"
        ),
    ]
)

Disc_question_reframer = Disc_reframe_prmpt | llm | StrOutputParser()

