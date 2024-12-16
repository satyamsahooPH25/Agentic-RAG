from json import load
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_core.tools import tool
import csv
import requests
import pandas as pd
import yfinance as yf
import math
import numexpr
import json
from typing import Dict
from urllib.request import urlopen
import certifi

from typing import Dict
import re

from typing import Optional

from collections import defaultdict
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from typing import List, Dict, Optional, Any
try:
    from newsapi import NewsApiClient
except:
    try:
        from newsapi.newsapi_client import NewsApiClient
    except:
        pass
from dotenv import load_dotenv

from urllib.request import urlopen
import certifi
import json


load_dotenv()


def request(url: str, method: str = "get", timeout: int = 10, **kwargs):
    """Helper to make requests from a url."""
    method = method.lower()
    assert method in [
        "delete",
        "get",
        "head",
        "patch",
        "post",
        "put",
    ], "Invalid request method."

    headers = kwargs.pop("headers", {})
    func = getattr(requests, method)
    return func(url, headers=headers, timeout=timeout, **kwargs)


def get_df(url: str, header: Optional[int] = None) -> pd.DataFrame:
    html = request(url).text
    # use regex to replace radio button html entries.
    html_clean = re.sub(r"(<span class=\"Fz\(0\)\">).*?(</span>)", "", html)
    return pd.read_html(html_clean, header=header)

api_key = {
    "ALPHA_VANTAGE": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "POLYGON": os.getenv("POLYGON_API_KEY"),
    "FINNHUB": os.getenv("FINNHUB_API_KEY"),
    "NEWSAPI": os.getenv("NEWSAPI_API_KEY"),
    "WOLFRAM": os.getenv("WOLFRAM_ALPHA_APPID"),
    "FINANCE": os.getenv("FINANCE_API_KEY")
}

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

web_search_tool = TavilySearchResults(include_raw_content=False,k=3)

duckduckgo_tool = DuckDuckGoSearchResults()

@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    try:
        wolfram = WolframAlphaAPIWrapper()
        print("calc")
        return str(wolfram.run(expression))
    except Exception as e:
        return f"Wolfram Alpha API failed with error: {e}"

@tool
def find_similar_companies(symbol: str, country=None):
    """
    Returns a list of companies similar to provided stock symbol.
    If country is None, performs a global search across all indices.
    """
    similar = []
    print("sim_comp")

    try:
        if "POLYGON" in api_key:
            result = request(
                f"https://api.polygon.io/v1/meta/symbols/{symbol.upper()}/company?&apiKey={api_key['POLYGON']}"
            )
            if result.status_code == 200:
                similar.extend(result.json()["similar"])
        if "FINNHUB" in api_key:
            result = request(
                f"https://finnhub.io/api/v1/stock/peers?symbol={symbol}&token={api_key['FINNHUB']}"
            )
            if result.status_code == 200:
                similar.extend(result.json())

        return similar
    
    except Exception as e:
        return f"Failed to find similar companies: {e}"


@tool
def get_earnings_history(symbol: str) -> pd.DataFrame:
    """
    Get actual, estimated earnings and surprise history for a given stock ticker symbol.

    If somehow api response is not found, returns an empty dataframe.
    """
    print("get_earn")
    try:
        earnings_df = pd.DataFrame()
        response = requests.request(
            "GET",
            f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key['ALPHA_VANTAGE']}",
        )
        if response.status_code != 200:
            return earnings_df

        print("Earnings EF")
        earnings_df = pd.json_normalize(response.json())

        earnings_df = pd.DataFrame(earnings_df["quarterlyEarnings"][0])
        earnings_df = earnings_df[
            [
                "fiscalDateEnding",
                "reportedDate",
                "reportedEPS",
                "estimatedEPS",
                "surprise",
                "surprisePercentage",
            ]
        ]
        return earnings_df.rename(
            columns={
                "fiscalDateEnding": "Fiscal Date Ending",
                "reportedEPS": "Reported EPS",
                "estimatedEPS": "Estimated EPS",
                "reportedDate": "Reported Date",
                "surprise": "Surprise",
                "surprisePercentage": "Surprise Percentage",
            }
        )
    except Exception as e:
        return f"Failed to get earnings history: {e}"


@tool
def get_latest_earning_estimate(symbol: str) -> float:
    """Gets latest actual and estimated earning estimate for a stock symbol."""

    print("get_lat_earn")
    try:
        df = yf.Ticker(symbol).earnings_dates
        return df["EPS Estimate"].loc[df["EPS Estimate"].first_valid_index()]
    except Exception as e:
        print(f"Failed to get latest earning estimate: {e}")
        try:
            print(f"Trying to get earnings history...")
            return get_earnings_history(symbol)
        except Exception as e:
            return f"Failed to get earnings history: {e}"


@tool
def get_upcoming_earnings(
    start_date: str,
    end_date: str,
    country: str,
    only_sp500: bool
):
    """Returns stocks announcing there earnings in next 3 months."""

    print("get_upcoming_earn")

    try:
        CSV_URL = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={api_key['ALPHA_VANTAGE']}"

        with requests.Session() as s:
            download = s.get(CSV_URL)
            decoded_content = download.content.decode("utf-8")
            cr = list(csv.reader(decoded_content.splitlines(), delimiter=","))
            df = pd.DataFrame(columns=cr[0], data=cr[1:])
            sd = pd.to_datetime(start_date, format="%Y-%m-%d")
            ed = pd.to_datetime(end_date, format="%Y-%m-%d")
            df["reportDate"] = pd.to_datetime(df["reportDate"])
            df = df[df["currency"] == country][df["reportDate"] > sd][df["reportDate"] < ed]
            if only_sp500:
                sp500 = pd.read_html(
                    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                )[0]
                sp500_tickers = list(sp500["Symbol"])
                df = df[df["symbol"].isin(sp500_tickers)]
            return df[["symbol", "name", "reportDate"]]
    except Exception as e:
        return f"Failed to get upcoming earnings: {e}"


@tool
def get_current_gainer_stocks() -> pd.DataFrame:
    """Return gainers of the day from yahoo finace including all cap stocks."""

    print("get-cur")
    try:
        # df_gainers = get_df("https://finance.yahoo.com/screener/predefined/day_gainers")[0]
        # df_gainers.dropna(how="all", axis=1, inplace=True)
        # return df_gainers.replace(float("NaN"), "")
        url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={api_key['FINANCE']}"

        return (get_jsonparsed_data(url))
    except Exception as e:
        return f"Failed to get current gainer stocks: {e}"


@tool
def get_current_loser_stocks() -> pd.DataFrame:
    """Get data for today's losers from yahoo finance including all cap stocks."""

    print("getcurlos")
    try:
        # df_losers = get_df("https://finance.yahoo.com/screener/predefined/day_losers")[0]
        # df_losers.dropna(how="all", axis=1, inplace=True)
        # return df_losers.replace(float("NaN"), "")
        url = f"https://financialmodelingprep.com/api/v3/stock_market/loosers?apikey={api_key['FINANCE']}"
        return (get_jsonparsed_data(url))
    except Exception as e:
        return f"Failed to get current loser stocks: {e}"


@tool
def get_current_undervalued_growth_stocks() -> pd.DataFrame:
    """Get data for today's stocks with low PR ratio and growth rate better than 25%."""
    print("getundervalue")
    try:
        df = get_df(
            "https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks"
        )[0]
        df.dropna(how="all", axis=1, inplace=True)
        return df.replace(float("NaN"), "")
    except Exception as e:
        return f"Failed to get current undervalued growth stocks: {e}"


@tool
def get_current_technology_growth_stocks() -> pd.DataFrame:
    """Get data for today's stocks with low PR ratio and growth rate better than 25%."""

    print("getcurtech")
    try:
        df = get_df(
            "https://finance.yahoo.com/screener/predefined/growth_technology_stocks"
        )[0]
        df.dropna(how="all", axis=1, inplace=True)
        return df.replace(float("NaN"), "")
    except Exception as e:
        return f"Failed to get current technology growth stocks: {e}"


@tool
def get_current_most_traded_stocks() -> pd.DataFrame:
    """Get data for today's stocks in descending order based on intraday trading volume."""
    print("getmosttraded")
    try:
        # Construct the API URL
        url = f"https://financialmodelingprep.com/api/v3/stock_market/actives?apikey={api_key['FINANCE']}"
        # Fetch and return the parsed JSON data
        return get_jsonparsed_data(url)
    except Exception as e:
        return f"Failed to get current most traded stocks: {e}"



@tool
def get_current_undervalued_large_cap_stocks() -> pd.DataFrame:
    """Get data for today's potentially undervalued large cap stocks from Yahoo finance."""

    print("getcurunder")
    try:
        df = get_df("https://finance.yahoo.com/screener/predefined/undervalued_large_caps")[
            0
        ]
        df.dropna(how="all", axis=1, inplace=True)
        return df.replace(float("NaN"), "")
    except Exception as e:
        return f"Failed to get current undervalued large cap stocks: {e}"


@tool
def get_current_aggressive_small_cap_stocks() -> pd.DataFrame:
    """Get data for today'sagressive / high growth small cap stocks from Yahoo finance."""

    print("aggres")
    try:
        df = get_df("https://finance.yahoo.com/screener/predefined/aggressive_small_caps")[
            0
        ]
        df.dropna(how="all", axis=1, inplace=True)
        return df.replace(float("NaN"), "")
    except Exception as e:
        return f"Failed to get current aggressive small cap stocks: {e}"


@tool
def get_current_hot_penny_stocks() -> pd.DataFrame:
    """Return data for today's hot penny stocks from pennystockflow.com."""

    print("penny")
    try:
        df = get_df("https://www.pennystockflow.com", 0)[1]
        return df.drop([10])
    except Exception as e:
        return f"Failed to get current hot penny stocks: {e}"


@tool
def get_current_stock_price_info(
 stock_ticker: str
) -> Optional[Dict[str, Any]]:
    """
    Return current price information given a stock ticker symbol.
    """

    print("stockinfo")
    try:
        result = request(
            f"https://finnhub.io/api/v1/quote?symbol={stock_ticker}&token={api_key['FINNHUB']}"
        )
        if result.status_code != 200:
            return f"Failed to get stock price info: {result.text}"
        return result.json()
    except Exception as e:
        return f"Failed to get stock price info: {e}"


@tool
def get_latest_news_for_stock(
 stock_id: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Returns latest news for a given stock_name by querying results via newsapi."""

    print("getlatestnews")
    try:
        newsapi = NewsApiClient(api_key=api_key['NEWSAPI'])
        cat_to_id = defaultdict(list)
        for source in newsapi.get_sources()["sources"]:
            cat_to_id[source["category"]].append(source["id"])
        business_sources = [
            "bloomberg",
            "business-insider",
            "financial-post",
            "fortune",
            "info-money",
            "the-wall-street-journal",
        ]
        for source in business_sources:
            assert source in cat_to_id["business"]

        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        articles = newsapi.get_everything(
            q=stock_id,
            sources=",".join(business_sources),
            from_param=start_date,
            to=end_date,
            language="en",
            sort_by="relevancy",
            page=1,
        )["articles"]
        return articles[:limit]
    
    except Exception as e:
        print(f"Could not fetch articles from get_latest_news_for_stock: {e}")
        try:
            print(f"Trying to fetch articles from get_topk_trending_news...")
            return get_topk_trending_news()
        except Exception as e:
            return f"Failed to fetch articles from get_topk_trending_news: {e}"
    


@tool
def get_topk_trending_news(
    limit: int = 10, extract_content: bool = True
) -> List[Dict[str, Any]]:
    """Returns top k trending news from seekingalpha."""

    print("topk")
    try:
        articles = []
        URL = "https://seekingalpha.com/news/trending_news"
        response = request(URL)
        if response.status_code == 200:
            for item in response.json():
                article_url = item["uri"]
                if not article_url.startswith("/news/"):
                    continue

                article_id = article_url.split("/")[2].split("-")[0]

                content = ""
                if extract_content:
                    article_url = f"https://seekingalpha.com/api/v3/news/{article_id}"
                    article_response = request(article_url)
                    jdata = article_response.json()
                    try:
                        content = jdata["data"]["attributes"]["content"].replace(
                            "</li>", "</li>\n"
                        )
                        content = BeautifulSoup(content, features="html.parser").get_text()
                    except Exception as e:
                        print(f"Unable to extract content for: {article_url}")

                articles.append(
                    {
                        "title": item["title"],
                        "publishedAt": item["publish_on"][: item["publish_on"].rfind(".")],
                        "url": "https://seekingalpha.com" + article_url,
                        "id": article_id,
                        "content": content,
                    }
                )
                if int(limit):
                    if len(articles) > limit:
                        break
                else:
                    if len(articles) > 10:
                        break
        print("articles length",len(articles))            
        return articles[:limit]
    except Exception as e:
        return f"Failed to get top trending news: {e}"


@tool
def get_google_trending_searches() -> Optional[pd.DataFrame]:
    """Returns overall trending searches in US unless region is provided."""
    # TODO(ishan): Can we filter by category?

    print("getggogletending")
    try:
        pytrend = TrendReq()
        return pytrend.trending_searches(pn="united_states")
    except Exception as e:
        return f"Unable to find google trending searches, error: {e}"


@tool
def get_google_trends_for_query(
    query: str, find_related: bool = False) -> Optional[pd.DataFrame]:
    """Find google search trends for a given query filtered by region if provided."""

    print("getgoodle\\")
    try:
        pytrend = TrendReq()
        # 12 is the category for Business and Industrial which covers most of the related
        # topics related to fin-gpt
        # Ref: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories

        # Only search for last 30 days from now
        pytrend.build_payload(
            kw_list=[query], timeframe="today 1-m", geo='', cat=12
        )
        return pytrend.interest_over_time()
    except Exception as e:
        return f"Unable to find google trend for {query}, error: {e}"


@tool
def get_latest_key_metrics(stock_id: str) -> Dict[str, str]:
    """Gives the primary key metrics of the given stock id"""
    print("I am here at get_latest_key_metrics")
    try:
        url = f"https://financialmodelingprep.com/api/v3/ratios/{stock_id}?apikey={api_key['FINANCE']}"
        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        final_json = json.loads(data)  # Parse the JSON data into a Python dictionary
        
        if not final_json:
            return []  # Return an empty list if no data is found

        final_json = final_json[0]  # Assuming the response is a list of dictionaries
        
        keys = {
            "priceEarningsRatio": "P/E Ratio",
            "priceBookValueRatio": "P/B Ratio",
            "debtEquityRatio": "Debt to Equity Ratio",
            "freeCashFlowPerShare": "Free Cash Flow",
            "priceEarningsToGrowthRatio": "PEG Ratio",
            "currentRatio": "Working Capital Ratio",
            "quickRatio": "Quick Ratio",
            "netProfitMargin": "Earnings Ratio",
            "returnOnEquity": "Return on Equity",
        }

        ratios = []
        for key, name in keys.items():
            value = final_json.get(key, None)  # Accessing the key from final_json, not from data
            ratios.append({"name": name, "value": value})  

        print("h1234521: ratios",ratios)
        return ratios
    
    except Exception as e:
        return f"Failed to get key metrics: {e}"


