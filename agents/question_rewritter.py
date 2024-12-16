### Question Re-writer
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=45,
    max_retries=2,
)

# LLM
# llm = ChatGroq(
#     model="llama-3.2-90b-vision-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )

# Prompt
system = """You are a question re-writer that processes input questions to a better version optimized for vectorstore retrieval.\nRewrite this question to make it clearer, more specific, and optimized for retrieving relevant information from a vectorstore.
    """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
# resp=question_rewriter.invoke({"question":"I am a stock market enthusiast and want to know about stpck status of company known by the name of apple"})
# print(resp)
