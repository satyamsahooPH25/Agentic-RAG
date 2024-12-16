from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


# Define the data model
class VerificationQuestions(BaseModel):
    """Data model for generated verification questions."""

    verification_questions: list = Field(
        description="List of questions for self-verification based on baseline responses."
    )


# LLM with structured output
# llm = ChatGroq(
#     model="llama-3.2-90b-text-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)

structured_llm = llm.with_structured_output(VerificationQuestions)

# Define the decomposition prompt
system_message = """
You are an expert in breaking down baseline responses into critical verification questions.
Your task is to generate essential and meaningful questions to verify the accuracy and completeness of a baseline response.

### Instructions:
1. Focus on importance: Avoid generating overly obvious or redundant questions. Instead, focus on key claims or facts that require verification.
2. Prioritize complexity: Generate questions that address the core differences, nuances, or assumptions in the baseline response.
3. Ensure relevance: Only ask questions that significantly impact the credibility or accuracy of the response.

### Example:

Query: "What are the key differences between a mutual fund and an exchange-traded fund (ETF)?"
Baseline Response: "Mutual funds and ETFs both pool investor money to purchase a diversified portfolio of assets, but they differ in how they are traded and managed. Mutual funds are actively managed, often leading to higher fees, and are priced at the end of the trading day. ETFs, on the other hand, are passively managed, have lower fees, and trade on stock exchanges throughout the day."

Verification Questions:
1. Do mutual funds consistently involve active management, or are there exceptions like passively managed index mutual funds?
2. Are the fees for mutual funds universally higher, or can they be comparable to certain ETFs?
3. Is it true that ETFs are typically passively managed, or are there actively managed ETFs in the market?
4. Does the frequency of trading for ETFs (throughout the day) significantly impact their liquidity and pricing compared to mutual funds?

Generate similar verification questions based on the given query and baseline response.
"""

decomposition_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "Query: {query}\nBaseline Response: {baseline_response}\n"),
    ]
)

verifier = decomposition_prompt | structured_llm
# query= "What are the key features of a Tesla Model S?"
# baseline_response="""The Tesla Model S is an electric car known for its long range, autopilot capabilities, and sleek design. "
#                     It has a range of up to 396 miles, accelerates from 0 to 60 mph in 3.1 seconds, and supports over-the-air updates."""
# for q in verifier.invoke({"query": query, "baseline_response": baseline_response}).verification_questions:
#     print(q)
