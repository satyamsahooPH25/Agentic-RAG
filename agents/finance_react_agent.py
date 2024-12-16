import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from tools.tools import (
    calculator,
    find_similar_companies,
    get_current_aggressive_small_cap_stocks,
    get_current_gainer_stocks,
    get_current_loser_stocks,
    get_current_most_traded_stocks,
    get_current_stock_price_info,
    get_current_technology_growth_stocks,
    get_current_undervalued_growth_stocks,
    get_current_undervalued_large_cap_stocks,
    get_earnings_history,
    get_google_trending_searches,
    get_google_trends_for_query,
    get_latest_news_for_stock,
    get_topk_trending_news,
    get_upcoming_earnings,
)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

tools = [
    find_similar_companies,
    get_earnings_history,
    get_upcoming_earnings,
    get_current_gainer_stocks,
    get_current_loser_stocks,
    get_current_undervalued_growth_stocks,
    get_current_technology_growth_stocks,
    get_current_most_traded_stocks,
    get_current_undervalued_large_cap_stocks,
    get_current_aggressive_small_cap_stocks,
    get_current_loser_stocks,
    get_topk_trending_news,
    get_google_trending_searches,
    get_google_trends_for_query,
    get_latest_news_for_stock,
    get_current_stock_price_info,
    calculator,
]

# llm = ChatGroq(
#     model="llama-3.2-90b-vision-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)

# Pulling prompt template from repo
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

finance_react_agent = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# agent_executor.invoke({"input": "Give me details about alphabet company. also future updates about it"})
