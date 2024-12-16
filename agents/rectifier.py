from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


# Define the data model for corrections
class CorrectedResponse(BaseModel):
    """Data model for corrected baseline response."""

    corrected_baseline: str = Field(
        description="Corrected version of the baseline response."
    )


# LLM with structured output
llm = ChatGroq(
    model="llama-3.2-90b-vision-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
structured_llm_correction = llm.with_structured_output(CorrectedResponse)

# Define the correction prompt
system_correction_message = """
You are an expert in verifying and refining baseline responses.
Your task is to correct the baseline response based on verified answers from self-verification questions. 

### Instructions:
1. Analyze the given baseline response.
2. Compare it with the verified answers provided.
3. Make corrections **only if there are significant discrepancies** in numerical figures, critical details, or drastic contradictions in the baseline response.
4. For minor discrepancies, retain the baseline response and add a disclaimer to clarify any potential inaccuracies or uncertain areas.
5. If the baseline response is mostly accurate and aligns with the verified answers, prioritize keeping it intact to ensure the user has some response to work with rather than an unclear or incomplete one.

### Formatting:
- For each change, ensure it is clearly integrated into the baseline response without introducing unnecessary ambiguity.
- Add disclaimers where needed to address areas where the information is uncertain or conflicting.

### Example:
Query: "What are the key differences between a mutual fund and an exchange-traded fund (ETF)?"

Baseline Response: 
"Mutual funds and ETFs both pool investor money to purchase a diversified portfolio of assets, but they differ in how they are traded and managed. Mutual funds are actively managed, often leading to higher fees, and are priced at the end of the trading day. ETFs, on the other hand, are passively managed, have lower fees, and trade on stock exchanges throughout the day."

Verified Answers:
1. Not all mutual funds are actively managed; some are passively managed.
2. Fees for mutual funds are generally higher, but there are low-cost mutual funds comparable to ETFs.
3. ETFs are mostly passively managed, but some actively managed ETFs exist.

Corrected Baseline Response:
"Mutual funds and ETFs both pool investor money to purchase a diversified portfolio of assets, but they differ in trading and management. While many mutual funds are actively managed, some are passively managed, similar to ETFs. Fees for mutual funds are typically higher but can sometimes be comparable to ETFs. Mutual funds are priced at the end of the trading day, while ETFs trade on stock exchanges throughout the day. ETFs are often passively managed, though actively managed ETFs are available."

---

Now, correct the baseline response using the following query, baseline response, and verified answers. If no significant changes are required, retain the baseline response but add disclaimers where necessary:
"""


correction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_correction_message),
        (
            "human",
            "Query: {query}\nBaseline Response: {baseline_response}\nVerified Answers: {verified_answers}",
        ),
    ]
)

rectifier = correction_prompt | structured_llm_correction

# query = "What are the key features of a Tesla Model S?"
# baseline_response = (
#     "The Tesla Model S is an electric car known for its long range, autopilot capabilities, and sleek design. "
#     "It has a range of up to 396 miles, accelerates from 0 to 60 mph in 3.1 seconds, and supports over-the-air updates."
# )
# verified_answers = [
#     "The Tesla Model S has a maximum range of 396 miles, but this depends on the driving mode and conditions.",
#     "Autopilot is a driver-assistance system, not full autonomous driving.",
#     "Acceleration from 0 to 60 mph in 3.1 seconds is specific to the Plaid model.",
# ]

# print(rectifier.invoke({
#         "query": query,
#         "baseline_response": baseline_response,
#         "verified_answers": verified_answers,
#     }))
