import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
)
from langchain_openai import ChatOpenAI
from tools.tools import (
    calculator,
    find_similar_companies,
    get_current_aggressive_small_cap_stocks,
    get_current_gainer_stocks,
    get_current_hot_penny_stocks,
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
    get_latest_key_metrics,
)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

tools = [
    find_similar_companies,
    get_earnings_history,#not working
    get_upcoming_earnings,#not working
    get_current_gainer_stocks,
    get_current_loser_stocks,
    get_current_undervalued_growth_stocks,
    get_current_technology_growth_stocks,
    get_current_most_traded_stocks,
    get_current_undervalued_large_cap_stocks,
    get_current_aggressive_small_cap_stocks,
    get_current_hot_penny_stocks,
    get_topk_trending_news,
    get_google_trending_searches,
    get_google_trends_for_query,
    get_latest_news_for_stock,
    get_current_stock_price_info,
    calculator,
    get_latest_key_metrics,
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


llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}


def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        print(tool_call["name"])
        tool_call["output"] = tool_map[tool_call["name"].lower()].invoke(
            tool_call["args"]
        )
    # print(tool_calls)
    return tool_calls


finance_agent = llm_with_tools | call_tools

# print(type(finance_agent.invoke("What are the trending Google searches in India?")[0]['output']))
# print(type(finance_agent.invoke("Can you find companies stock similar to Apple (AAPL) in the US market?")[0]['output']))
# print(finance_agent.invoke("What is the earnings history for Google (GOOGL)?")[0]['output'])#<class 'str'>#Failed to get
# print(finance_agent.invoke("What is the latest earnings estimate for Microsoft (MSFT)?")[0]['output'])#"Failed to get" 
# print(finance_agent.invoke("Which S&P 500 companies have earnings announcements between November 1 and November 30, 2024, in USD?"))#"Failed to get"
# print(finance_agent.invoke("Can you list today's top gainer stocks ?")[0]['output'])#<class 'pandas.core.frame.DataFrame'>
# print(type(finance_agent.invoke("Can you list today's top looser stocks?")[0]['output']))#<class 'list'> empty else dataframe # DOUBTFUL
# print("Will be right back" in finance_agent.invoke("Can you find undervalued growth stocks for today with low P/E ratios?")[0]['output'][0][0])#"Will be right back"
# print(finance_agent.invoke("What are today's top technology growth stocks?")[0]['output'][0])
# print(finance_agent.invoke("Which stocks have the highest trading volume today?")[0]['output'])
# print(finance_agent.invoke("Can you show me undervalued large-cap stocks for today?")[0]['output'][0][0])#"Will be right back"
# print(finance_agent.invoke("What are today's high-growth small-cap stocks?")[0]['output'][0][0])#"Will be right back"
# print(finance_agent.invoke("Can you provide today's hot penny stocks?")[0]['output'])
# print(finance_agent.invoke("What is the current stock price information for Tesla (TSLA)?")[0]['output'])
# print(finance_agent.invoke("What are the latest news articles about Meta (META)?")[0]['output'])
# print(finance_agent.invoke("What are the top 5 trending financial news articles today?")[0]['output'])
# print(finance_agent.invoke("What are the trending Google searches in India?")[0]['output'])
print(finance_agent.invoke("What are the key metrics of Google ?")[0]['output'])

