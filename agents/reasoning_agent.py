### Question Re-writer
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

# LLM
# llm = ChatGroq(
#     model="llama-3.2-90b-text-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )



# Prompt
system = """You are a good reasoner. You are given a question and some of the information that may be relevant for answering the question.
Your task is to reason and give the best context that is needed to answer the question. In simpler terms arrange the data in given in the context in more interpretable manner. If there's a need to create chart, do specify it using chart creation or similar keywords. Don't hallucinate!"""

reason_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Here is the context: \n\n {documents} \n Formulate an good context.",
        ),
    ]
)

reasoner = reason_prompt | llm | StrOutputParser()
# resp=reasoner.invoke({"question":"What are the benefits of using green energy in residential homes?","documents":"In recent years, there's been a growing emphasis on sustainable living, particularly in urban neighborhoods. One of the key shifts has been towards renewable sources of energy for powering homes. Solar panels, for example, have become more affordable and are now seen on rooftops in various regions. Using solar energy reduces electricity bills significantly, making it a cost-effective solution over the long term. Wind turbines are another option, particularly in areas with consistent winds, providing a clean alternative to fossil fuels.In addition to cost savings, green energy sources like solar and wind have a lower environmental impact compared to traditional energy sources. They contribute to reducing carbon emissions, which plays a role in mitigating climate change. Government incentives and tax rebates have also motivated homeowners to make the switch to renewable energy. This move not only lowers individual carbon footprints but also supports energy independence. For those concerned about reliability, modern battery storage solutions are available to store excess energy for use during cloudy days or at night."})
# print(resp)
