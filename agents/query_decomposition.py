### Hallucination Grader

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)

# Data model
class QueryDecomposition(BaseModel):
    """to get List of subqueries"""

    sub_queries: list = Field(
        description="List of sub-queries only decomposed from the original query."
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
structured_llm_grader = llm.with_structured_output(QueryDecomposition)

# Prompt
system = """You are a Query Decomposition Specialist for a Financial Information Retrieval System.

CORE OBJECTIVE:
Strategically analyze and process complex financial queries.

DECOMPOSITION PROCESS:

STEP 1: QUERY EVALUATION
Determine if the query requires decomposition by checking:
- Is the query multi-dimensional?
- Does it involve complex financial analysis?
- Are multiple perspectives needed?

STEP 2: QUERY PROCESSING
A. IF Decomposition is NOT NEEDED:
   - Return the original query

B. IF Decomposition is REQUIRED:
   - Generate 2-3 precise, non-overlapping sub-queries
   - Each sub-query must:
     * Be specific
     * Cover a distinct aspect of the original query
     * They should not depend on each other
     * Enable targeted information retrieval

GUIDING PRINCIPLES:
- Maintain query's original intent
- Prioritize precision over complexity
- Don't create redundant sub-queries

Query: "{{input_query.query}}"

Provide the sub-queries as a list of concise and distinct strings only.
"""

decomposition_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Query: \n\n {question}"),
    ]
)

decomposer = decomposition_prompt | structured_llm_grader

# Example
# query = "Can you find companies stock similar to Apple (AAPL) in the US market?"
# ans = decomposer.invoke({"question":query})
# print(ans)
