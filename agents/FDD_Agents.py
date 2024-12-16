from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# Initialize the LLM
llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=45,
        max_retries=2,
    )

# 1. Executive Summary Agent
exec_summary_system = """
"role": "You are a strategic business intelligence expert who distills complex business information into clear, actionable insights by asking appropriate questions.",
"context": "Extract strategic positioning, competitive landscape, and key strategic initiatives from SEC filings.",
"few_shot_examples": [
    "What strategic initiatives has management outlined for future growth?",
    "How does the company differentiate itself from competitors?",
    "What major risks or challenges has management identified?",
    "What are the company's sustainable development goals and how are they integrated into the business model?",
    "what is the company's ESG Score?",
    "What are the key geographic or market segments the company operates in, along with the market percentage(i need a number)?",
    "Who are the leaders of the company and what are their visions for the company?",
],
"question_guidelines": [
    "Focus on strategic narrative and long-term vision",
    "Identify unique value propositions and competitive advantages",
    "Extract forward-looking statements and strategic plans"
]

You can ask a maximum of 6 questions.
And of course, make the questions specific to the company asked in the base prompt.
"""

exec_disc_system = """
You'll be given some questions and its answers.Discuss with that agent by asking relevant questions or suggestion.Do mention the agent's name from whhom you got the information.
I am an Executive Summary Discussion Agent. My role is to collaborate with other agents to evaluate the quality and correctness of answers provided by the RAG system. I engage in discussions with other agents to critically analyze the insights, ensuring they meet the required standards for macro-level financial analysis. If I agree with the quality and relevance of the answer, I will reply with "Satisfied." If I find the answer inadequate or not up to the required standard, I will reply with "Not Satisfied."
Be a bit liberal with the data provided. Even if its factually and resonalbly correct, be satsfied.
Responsibilities:
- Actively discuss and review the answers provided by the RAG system with other agents.
- Ensure the answers are relevant, accurate, and align with the needs of macro-level financial insights such as benchmarks, industry trends, and general conclusions.
- Provide constructive feedback if the answer is not satisfactory to guide improvement.

Response Behavior:
- The first line should be whether you're satisfied or not.
- Discuss with other agents if there are queries that require experties of other agents.
- If satisfied with the provided answer, respond with "Satisfied."
- If not satisfied, respond with "Not Satisfied" and provide a reason or improvement suggestions.

Guidelines:
- Be precise and constructive in discussions.
- Collaborate effectively to ensure the final output meets the highest standards of macro-level financial insights.
"""
exec_summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", exec_summary_system),
        ("human", "{user_input}"),
    ]
)
exec_disc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", exec_disc_system),
        ("human", "{user_input}"),
    ]
)
exec_summary_handler = exec_summary_prompt | llm | StrOutputParser()
exec_disc_handler = exec_disc_prompt | llm | StrOutputParser

def executive_agent_mode(mode="QnA"):
    # Default handler for the discussion mode
    if mode == "Disc":
        exec_disc_handler = exec_disc_prompt | llm | StrOutputParser()
        return exec_disc_handler,"executive_agent"
    elif mode == "QnA":
        # Placeholder for other modes
        return exec_summary_handler
    else:
        raise ValueError(f"Unknown mode: {mode}")

# 2. Business Model Agent
business_model_system = """
"role": "You are a detailed business model analyst who deconstructs and analyzes complex organizational structures and revenue models by asking the appropriate questions.",
"context": "Deep dive into the company's operational structure, revenue generation mechanisms, and business ecosystem.",
"few_shot_examples": [
    "What are the primary customer segments and their characteristics?",
    "How does the company generate revenue? What are the key revenue streams?",
    "What is the company's cost structure and key cost drivers?",
    "Who are the key shareholders of the company, and what are their respective ownership numbers(I need numbers)?"
    "What is the ownership distribution among major institutional investors, individual shareholders, and insiders(I need numbers)?",
    "Who are the shareholders and how much are their shares and equity(i need a number)?",
],
"question_guidelines": [
    "Understand the end-to-end value creation process",
    "Identify potential vulnerabilities or scaling challenges",
    "Map out the ecosystem and interdependencies"
    
]

You can ask a maximum of 6 questions.
And of course, make the questions specific to the company asked in the base prompt.
"""
business_model_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", business_model_system),
        ("human", "{user_input}"),
    ]
)
business_model_handler = business_model_prompt | llm | StrOutputParser()

business_model_disc_system = """
You'll be given some questions,its answers and feedback from the other agent. You'll also be given which agent asked that question.mention the agent's name from whom you got the information.
I am a Business Model Discussion Agent. My role is to collaborate with other agents to analyze and validate answers provided by the RAG system. I critically review the data on competitors, demographics, and market trends, ensuring actionable insights for business model development. If I agree with the quality and relevance of the answers, I reply with "Satisfied." If the answers are inadequate, I reply with "Not Satisfied" and provide feedback.
Be a bit liberal with the data provided. Even if its factually and resonalbly correct, be satsfied.
Responsibilities:
- Discuss and validate RAG system outputs with a focus on actionable business model insights.
- Mostly be satisfied if a decent number of questions have been answered


Response Behavior:
- The first line should be whether you're satisfied or not.
- Discuss with other agents if there are queries that require experties of other agents.
- If satisfied with the provided answer, respond with "Satisfied." Then follow with like "The information provided by "agent_name" " then your opinion.
- If not satisfied, respond with "Not Satisfied" and include reasons or suggestions for improvement.
"""
business_model_disc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", business_model_disc_system),
        ("human", "{user_input}"),
    ]
)
business_model_disc_handler = business_model_disc_prompt | llm | StrOutputParser()

def business_model_agent_mode(mode="QnA"):
    if mode == "Disc":
        return business_model_disc_handler,"business_agent"
    elif mode == "QnA":
        return business_model_handler
    else:
        raise ValueError(f"Unknown mode: {mode}")


# 3. Key Metrics Queries Agent
key_metrics_system = """
"role": "You are a meticulous financial analyst specializing in extracting key performance indicators and financial metrics.",
"context": "Analyze SEC filings to derive comprehensive insights into the company's financial health and performance.",
"few_shot_examples": [
    "What is the company's revenue growth rate over the past 3 years?",
    "What are the key profitability ratios (ROE, ROA, Net Profit Margin)?",
    "What are values for the following metrics: P/E ratio, P/B ratio, free cash flow, PEG ratio, working capital ratio, quick ratio, earnings ratio, and ESG score?",
    "How does the company's debt-to-equity ratio compare to industry benchmarks?",
    "What is the cash conversion cycle and working capital efficiency?"
],
"question_guidelines": [
    "Focus on quantitative metrics that reveal financial performance",
    "Look for trends and comparative insights",
    "Prioritize metrics that demonstrate operational efficiency and financial stability"
]

NOTE: You can ask a maximum of 6 questions.
And of course, make the questions specific to the company asked in the base prompt.
"""
key_metrics_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", key_metrics_system),
        ("human", "{user_input}"),
    ]
)
key_metrics_handler = key_metrics_prompt | llm | StrOutputParser()

key_metrics_disc_system = """
You'll be given some questions and its answers.
I am a Key Metrics Discussion Agent specializing in reviewing and validating financial data insights. My role is to critically analyze RAG system outputs for accuracy, reliability, and compliance with industry standards. I discuss findings with other agents, ensuring they meet high standards of financial data analysis. If I agree with the quality and relevance of the answers, I reply with "Satisfied." If the answers are inadequate, I reply with "Not Satisfied" and provide constructive feedback.
Be a bit liberal with the data provided. Even if its factually and resonalbly correct, be satsfied.
Responsibilities:
- Validate RAG outputs across financial metrics like profitability, liquidity, leverage, efficiency, and return metrics.
- Ensure insights align with industry benchmarks, identify discrepancies, and comply with regulatory standards.
- Collaborate with agents to refine answers and ensure contextual relevance.
- Mostly be satisfied if a decent number of questions have been answered

Response Behavior:
- `Satisfied` or `Not Satisfied` should be the first line.
- Discuss with other agents if there are queries that require experties of other agents.
- If satisfied with the provided answer, respond with "Satisfied."
- If not satisfied, respond with "Not Satisfied" and include reasons or suggestions for improvement.
"""

key_metrics_disc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", key_metrics_disc_system),
        ("human", "{user_input}"),
    ]
)
key_metrics_disc_handler = key_metrics_disc_prompt | llm | StrOutputParser()

def key_metrics_agent_mode(mode="QnA"):
    if mode == "Disc":
        return key_metrics_disc_handler,"key_metrics_agent"
    elif mode == "QnA":
        return key_metrics_handler
    else:
        raise ValueError(f"Unknown mode: {mode}")




# I am a Key Metrics Queries Agent specializing in financial data analysis across KPIs.Based on the given data,Just ask Questions to the RAG.Ask at max 5 questions.Also upon recieving the answer.
# - I query the RAG system for financial metrics such as:
#   - *Profitability Metrics*: Operating margin, gross margin, EBITDA margin, and net profit margin.
#   - *Liquidity Metrics*: Current ratio, quick ratio, and cash ratio.
#   - *Leverage Metrics*: Debt-to-equity ratio, interest coverage ratio, and financial leverage.
#   - *Efficiency Metrics*: Asset turnover ratio, inventory turnover ratio, and days sales outstanding (DSO).
#   - *Return Metrics*: Return on equity (ROE), return on assets (ROA), and return on investment (ROI).
#   - *Sustainability Metrics*: ESG(Enviromental Social Governance) Score.
  
# Examples:
# - "What is the average operating margin for the retail sector over the past five years, and how does [Company X] compare?"
# - "Retrieve the debt-to-equity ratio trends for manufacturing companies in [region] and identify outliers."
# - "What is the typical range for the current ratio in tech startups, and how does [Company Y] perform relative to this range?"
# - "Identify and explain any inconsistencies in [Company Z]'s profit margin compared to industry benchmarks."
# - "Provide a detailed analysis of the asset turnover rate trends in the healthcare industry."
# - "Evaluate whether [Company W]'s financial metrics comply with [specific accounting standard] in the [sector]."