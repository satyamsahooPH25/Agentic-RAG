import json
from textwrap import dedent
from openai import OpenAI
from pydantic import BaseModel
from typing import List
client = OpenAI()

MODEL = "gpt-4o"

dashboard_prompt = '''
    You are an AI trained to provide detailed and structured information in response to business and financial queries. 
    Please provide the requested data in JSON format where applicable, using the appropriate structure for each section.
'''

class KeyMetric(BaseModel):
    p_e_ratio: str
    p_b_ratio: str
    debt_to_equity_ratio: str
    free_cashflow: str
    peg_ratio: str
    working_capital_ratio: str
    quick_ratio: str
    earning_ratio: str
    return_on_equity: str
    esg_score: str

class RevenueItem(BaseModel):
    revenue_source: str
    revenue_value: int

class Market(BaseModel):
    country: str
    market_percentage: str

class Management(BaseModel):
    name: str
    designation: str
    vision_for_company: str

class SDG(BaseModel):
    sdg_number: str
    goal_description: str
    contribution: str

class Metrics(BaseModel):
      metrics: List[KeyMetric]

class RevenueSources(BaseModel):
    revenue: List[RevenueItem]

class Markets(BaseModel):
    countries: List[Market]

class Managements(BaseModel):
    team: List[Management]

class SDGs(BaseModel):
    goals: List[SDG]




def get_json(context:str):
    answer = []

    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                Can you provide the following financial metrics:  
                1. Price-to-Earnings (P/E) Ratio  
                2. Price-to-Book (P/B) Ratio  
                3. Debt-to-Equity Ratio  
                4. Free Cash Flow  
                5. Price/Earnings-to-Growth (PEG) Ratio  
                6. Working Capital Ratio  
                7. Quick Ratio  
                8. Earnings Per Share (EPS)  
                9. Return on Equity (ROE)  
                10. ESG Score  
                ''')},
        ],
        response_format=Metrics,
    )
    answer.append(completion.choices[0].message.content)

    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                "Can you provide the following details for the company, including the following details for each revenue source(keep the units consistent for each source):
                    1. Revenue Source Name
                    2. Value of Revenue
                ''')},
        ],
        response_format=RevenueSources,
    )
    answer.append(completion.choices[0].message.content)

    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                Can you provide the market penetration data for the company in different countries? give the country name in geojson NAME format(refer your LLM knowledge base for location geography via geojson dataset from as latest updates as possible) 
                ie, give United States of America instead of United States and so on:
                    1. Country name
                    2. Market penetration percentage
                ''')},
        ],
        response_format=Markets,
    )
    answer.append(completion.choices[0].message.content)    

    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                Can you provide details about the management team and their ownership in the company? Include the following information for each member:
                    1.  Name of the person
                    2. Designation (e.g., CEO, CTO, CFO)
                    3. Vision for the company
                ''')},
        ],
        response_format=Managements,
    )
    answer.append(completion.choices[0].message.content)


    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(dashboard_prompt)},
            {"role": "user", "content": (
                context + '''
                "Can you provide details about the company's contributions to the UN Sustainable Development Goals (SDGs)? Include the following information for each SDG the company supports:
                    1. SDG number
                    2. Goal description
                    3. Contribution by the company
                ''')},
        ],
        response_format=SDGs,
    )
    answer.append(completion.choices[0].message.content)


    merged_dict = {}
    for json_str in answer:
      data = json.loads(json_str)  
      merged_dict.update(data)    


    # output_file = "merged_data.json"
    # with open(output_file, "w") as file:
    #   json.dump(merged_dict, file, indent=4)

    return merged_dict



context = '''
    Here’s a set of dummy data in text format for each of the sections based on the previously generated questions:
---
### Capitalization Table
**Capitalization Table:**
- **Founder 1**:  
  Shares: 500,000  
  Ownership: 50%
- **Founder 2**:  
  Shares: 300,000  
  Ownership: 30%
- **Investor A**:  
  Shares: 100,000  
  Ownership: 10%
- **Employee Pool**:  
  Shares: 50,000  
  Ownership: 5%
- **Investor B**:  
  Shares: 50,000  
  Ownership: 5%
---
### Market Penetration
**Market Penetration:**
- **United States**:  
  Market Penetration: 75%
- **Canada**:  
  Market Penetration: 60%
- **Germany**:  
  Market Penetration: 45%
- **India**:  
  Market Penetration: 30%
- **Brazil**:  
  Market Penetration: 40%
- **Japan**:  
  Market Penetration: 50%
- **Australia**:  
  Market Penetration: 70%
- **South Africa**:  
  Market Penetration: 20%
---
### Management Ownership and Vision
**Management Ownership and Vision:**
- **John Doe (CEO)**:  
  Vision: To lead the company to become the global leader in innovative technology, focusing on sustainability and market disruption.
- **Jane Smith (CTO)**:  
  Vision: To create cutting-edge, scalable technologies that redefine user experiences and drive industry growth.
- **Michael Johnson (CFO)**:  
  Vision: To build strong financial foundations and ensure sustainable growth, focusing on long-term profitability and shareholder value.
---
### SDG Contributions
**SDG Contributions:**
- **SDG 2 (Zero Hunger)**:  
  Contribution: Through partnerships with food banks and sustainable farming initiatives, the company helps reduce hunger and food insecurity.
- **SDG 3 (Good Health and Well-being)**:  
  Contribution: The company promotes employee health and well-being with comprehensive health programs and supports access to healthcare in underdeveloped regions.
- **SDG 4 (Quality Education)**:  
  Contribution: The company invests in education by providing scholarships and training programs for underserved communities and employees.
- **SDG 5 (Gender Equality)**:  
  Contribution: The company fosters a diverse and inclusive workplace, actively promoting gender equality in hiring and leadership roles.
- **SDG 7 (Affordable and Clean Energy)**:  
  Contribution: The company invests in renewable energy sources and works on developing energy-efficient products.
- **SDG 9 (Industry, Innovation, and Infrastructure)**:  
  Contribution: The company supports sustainable industry practices, innovation in technology, and infrastructure development in emerging markets.
- **SDG 12 (Responsible Consumption and Production)**:  
  Contribution: The company adheres to sustainable sourcing practices, minimizes waste, and promotes recycling and circular economy initiatives.
- **SDG 13 (Climate Action)**:  
  Contribution: The company has committed to reducing its carbon footprint through energy-efficient operations, carbon offset programs, and sustainable product designs.
- **SDG 17 (Partnerships for the Goals)**:  
  Contribution: The company collaborates with NGOs, governments, and other businesses to achieve SDG targets through shared initiatives and joint projects.
    '''

# json_dict = get_json(context)
# print(json_dict)