from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)

# LLM for Chart Descriptions
# llm = ChatGroq(
#     model="llama-3.2-90b-text-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# Prompt for Chart Creation
chart_system = """
You are an expert data visualization assistant. Your task is to take a context and decide if it can be visualized as a chart. 
If possible, generate a JSON representation of the data for chart creation. 

The JSON should contain:
- *chartType*: The type of chart (e.g., "bar", "line", "pie").
- *title*: Title of the chart.
- *xAxisLabel*: Label for the X-axis (applicable for bar and line charts).
- *yAxisLabel*: Label for the Y-axis (applicable for bar and line charts).
- *data*: A list of objects where each object has:
  - *label*: The category or X-axis label.
  - *value*: The numerical value.
- For pie charts, include:
  - *labels*: A list of category names.
  - *values*: A list of numerical values corresponding to each label.
  - *colors*: (Optional) A list of colors for the pie sections.
  - *legend*: (Optional) An object with:
    - *display*: A boolean to indicate if the legend should be displayed.
    - *position*: The position of the legend (e.g., "best", "upper right").
  
If the context cannot be visualized as a chart, respond with "No chart possible."
"""

chart_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chart_system),
        (
            "human",
            "Here is the context: \n\n {generation} \n If it can be visualized as a chart, return a JSON representation of the data and chart type. Otherwise, say 'No chart possible.'",
        ),
    ]
)

chart_generator = chart_prompt | llm | StrOutputParser()