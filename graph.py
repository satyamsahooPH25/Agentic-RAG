import os
import streamlit as st
from pprint import pprint
import sys
import io
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.callbacks import get_openai_callback
import matplotlib.pyplot as plt
import json
import os
import re
import time
import logging
from pprint import pprint
from typing import List
from tools.helper import MoveCharts
from guardrails.llamaguard import llamaguard
from guardrails.PII import pii_remover

def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


load_environment_variables()

from agents.Chart_Agent import chart_generator
from agents.SQLAgent import SQLAgent
from agents.TableMaker import TableMaker
from agents.FDD_Agents import executive_agent_mode,key_metrics_agent_mode,business_model_agent_mode
from agents.FDD_generator import FDD_Generator_handeler
from agents.visual_json import get_json
from agents.Disc_reframer import Disc_question_reframer
from agents.answer_aggregator import aggregator
from agents.answer_grader import answer_grader
from agents.document_relevent_router import relevency_router
from agents.finance_react_agent import finance_react_agent
from agents.finance_agent import finance_agent
from agents.hallucination_grader import hallucination_grader
from agents.main_router import question_router
from agents.query_decomposition import decomposer
from agents.question_rewritter import question_rewriter
from agents.reasoning_agent import reasoner
from agents.rectifier import rectifier
from agents.retrieval_grader import retrieval_grader
from agents.verification_agent import verifier
from langchain.schema import Document
from langgraph.graph import END, START, StateGraph
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from rag.rag import (
    compress_documents,
    create_compressor,
    create_groq_llm,
    create_openai_llm,
    create_pathway_client,
    create_prompt_template,
    get_answer,
    retrieve_relevant_documents,
)
from tools.bar_maker import generate_bar_chart
from tools.line_maker import generate_line_chart
from tools.pie_maker import generate_pie_chart
from tools.tools import web_search_tool
from tools.tools import duckduckgo_tool
from typing_extensions import TypedDict

llm = create_openai_llm()
prompt = create_prompt_template()
BASELINE_VERIFICATION_QUESTIONS = []

# need to add this as a state value in workflow, havent done it yet
NOT_SUPPORTED_COUNTER = 0
WEB_SEARCH_COUNTER = 0

# Redirecting stdout to capture print statements (for logging)
class StreamToLogger(io.StringIO):
    def __init__(self):
        super().__init__()
        self.log = ""
    
    def write(self, message):
        self.log += message + "\n"  # Add newline to each log message for clarity
        sys.__stdout__.write(message)  # Also write to the original stdout
    
    def get_logs(self):
        logs = self.log.strip()  # Remove any leading/trailing whitespace
        self.clear_logs()  # Clear logs after retrieval to prevent duplication
        return logs
    
    def clear_logs(self):
        self.log = ""  # Clear stored logs

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)    

logger_stream = StreamToLogger()
sys.stdout = logger_stream  # Redirects print statements to this logger

        
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        max_revision: int
    """

    question: str
    generation: str
    documents: List[str]
    revision_number: int
    max_revisions: int
    final_generation: str
    first_attempt: bool

tokens=0
inp_tok=0

async def retrieve(state):
    """
    Retrieve documents based on the current state.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with the key "documents" containing retrieved and compressed documents.
    """
    if "revision_number" not in state or "question" not in state:
        raise ValueError("State must contain 'revision_number' and 'question' keys.")

    question = state["question"]

    # Initialize token tracking
    global tokens
    global inp_tok
    tok = 0
    pr_tok = 0
    try:

        # if state.get("revision_number", 0) == 0:
        print("---RETRIEVE---")
        #print("Question",state["question"])
        # Retrieval
        retriever = await create_pathway_client()
        compressor = create_compressor()

        tok, pr_tok, compressed_docs = compress_documents(retriever, question, compressor)
        tokens += tok
        inp_tok += pr_tok
        print("Len Docs:",len(compressed_docs))
        if not compressed_docs:
            print("No documents retrieved or compressed.")

        return {
            "documents": compressed_docs,
            "question": question,
            # "revision_number": state["revision_number"] + 1,
            "revision_number": state["revision_number"],
            "max_revisions": state["max_revisions"],
            "first_attempt": state["first_attempt"]
        }

    except:
        return web_search(state)


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    print("Question",state["question"])
    global tokens, inp_tok

    question = state["question"]
    documents = state["documents"]
    print(f"In GETANSWER {documents}")

    final_answer_pairs = ""
    for doc in documents:
        try:
            final_answer_pairs += "###Document::\n" + f'content: {doc.page_content} || doc name :{doc.metadata["name"]} || page number: {doc.metadata["page_number"]}'
        except:
            final_answer_pairs += "###Document::\n" + f'content: {doc.page_content} || metadata: {doc.metadata}'
        
        
    tok, pr_tok, generation = get_answer(final_answer_pairs, question, llm, prompt)
    tokens += tok
    inp_tok += pr_tok
    print("Generate node output: ", generation)
    print(f"Total tokens: {tok}")
    print(f"Prompt tokens: {pr_tok}")
    print("First Generation: ", state["first_attempt"])

    return {
        "documents": documents,
        "question": question,
        "final_generation": generation,
        "revision_number": state["revision_number"],
        "max_revisions": state["max_revisions"],
        "first_attempt": state["first_attempt"]
    }



def grade_documents(state):
    global tokens
    global inp_tok
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    print("Question",state["question"])
    question = state["question"]
    documents = state["documents"]

    # Score each document
    filtered_docs = []
    for doc in documents:
        with get_openai_callback() as cb:
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens
        print(cb.total_tokens)
        print(cb.prompt_tokens)
        
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    if state["revision_number"] == state["max_revisions"]:
        print("----MAXIMUM REVISIONS REACHED, NOT FILTERING DOCUMENTS----")
        return {"documents": documents, "question": question}

    return {
        "documents": filtered_docs,
        "question": question,
        "revision_number": state["revision_number"],
        "max_revisions": state["max_revisions"],
        "first_attempt": state["first_attempt"]
    }


def transform_query(state):
    global tokens
    global inp_tok
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    print("Question",state["question"])
    question = state["question"]
    documents = state["documents"]

    # Re-write the question
    with get_openai_callback() as cb:
        better_question = question_rewriter.invoke({"question": question})
        state["revision_number"] += 1

    # Update token counts
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens

    print(f"Total tokens: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    
    if state["first_attempt"] == True:
        state["first_attempt"] = False

    return {
        "documents": documents,
        "question": better_question,
        "revision_number": state["revision_number"],
        "max_revisions": state["max_revisions"],
        "first_attempt": state["first_attempt"]
    }


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]
    state["documents"]=state.get("documents",[])
    
    for doc in state["documents"]:
        if doc.metadata.get("source")=="web_search":
           return state 
    
    print("---WEB SEARCH---")
    print("Question",state["question"])
        
    # Perform web search
    try:
        search_results = web_search_tool.invoke({"query": question})
    except Exeption as e:
        print(f"Tavily Web search failed...{e}")
        try:
            print("Using DuckDuck go search...")
            search_results = duckduckgo_tool.invoke(question)
            search_results = [{"content": result} for result in search_results]
            
        except Exception as e:
            print(f"DuckDuckGo search failed...{e}")
            search_results = []
    
    # Combine search results into a single document
    try:
        combined_results = "\n".join([result["content"] for result in search_results])
    except Exception as e:
        print(f"Error combining search results: {e}")
        print("Search results: ", search_results)
        combined_results = ""
    web_document = Document(page_content=combined_results, metadata={"source": "web_search"})
    state["documents"].extend([web_document])
    # print("DOCUMENTS-WEB_SEARCH",state["documents"])
    global WEB_SEARCH_COUNTER
    WEB_SEARCH_COUNTER += 1
    return {
        "documents": state["documents"], 
        "question": question,
        "revision_number": state["revision_number"],
        "max_revisions": state["max_revisions"],
        "first_attempt": state["first_attempt"]
    }


### Edges ###


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    print("Question",state["question"])
    global tokens, inp_tok
    question = state["question"]

    # Determine routing using question_router with token tracking
    with get_openai_callback() as callback:
        routing_result = question_router.invoke({"question": question})
        
        # Update token counters
        tokens += callback.total_tokens
        inp_tok += callback.prompt_tokens
        
        # Log token usage
        print(f"Total tokens: {callback.total_tokens}")
        print(f"Prompt tokens: {callback.prompt_tokens}")

    # Route based on determined data source
    if routing_result.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    else:  # vectorstore case
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_retrieve(state):
    """
    Determines whether to retrieve a question (compressed or without compression handled inside retriever) 
    or web-search
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """
    print("---Decide To Review---")
    print("Question",state["question"])
    if state["revision_number"] <= 1:
        return "retrieve"
    else:
        return "web_search"
    
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, re-generate a question, 
    or route to financial or SQL agents based on the document structure.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    global tokens
    global inp_tok
    print("---ASSESS GRADED DOCUMENTS---")
    print("Question",state["question"])
    question = state["question"]
    filtered_documents = state["documents"]
    # print("filtered documents: ", filtered_documents)
    filtered_documents = [doc.page_content for doc in filtered_documents]
    
    with get_openai_callback() as cb:
        answerability = relevency_router.invoke({"question": question, "context": filtered_documents})
    
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    
    print(cb.total_tokens)
    print(cb.prompt_tokens)
    print("Answerability: ", answerability)
    
    if filtered_documents and answerability.datasource != "answerable" and state["first_attempt"]==False:
        print("---DECISION: REDIRECTING TO FINANCE AGENT ---")
        return "not_answerable"

    # Check if all documents are filtered out and if more revisions are allowed
    if not filtered_documents and state["revision_number"] < state["max_revisions"]:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"

    # Check for multiple tables in "text_as_html"
    
    total_table_count = 0
    for doc in filtered_documents:
        # st.write(doc)
        if "text_as_html" in doc:
            soup = BeautifulSoup(doc["text_as_html"], "html.parser")
            total_table_count += len(soup.find_all("table"))

    if total_table_count >= 2:
        print("---DECISION: MULTIPLE TABLES DETECTED, ROUTING TO SQL_AGENT---")
        return "sql_agents"

    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"


def grade_generation_v_documents_and_question(state):
    global tokens
    global inp_tok
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    print("Question",state["question"])
    global tokens, inp_tok
    question = state["question"]
    documents = state["documents"]
    documents_content = [doc.page_content for doc in documents]
    generation = state["final_generation"]

    # Check for hallucinations
    with get_openai_callback() as cb:
        score = hallucination_grader.invoke({"documents": documents_content, "generation": generation})
        hallucination_grade = score.binary_score
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    print(cb.total_tokens)
    print(cb.prompt_tokens)

    if hallucination_grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        # Check if the generation answers the question
        print("---GRADE GENERATION vs QUESTION---")
        with get_openai_callback() as cb:
            score = answer_grader.invoke({"question": question, "generation": generation})
            answer_grade = score.binary_score
        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens
        print(cb.total_tokens)
        print(cb.prompt_tokens)

        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if state["first_attempt"] == True:
                return "not useful"
                
            if state["revision_number"] >= state["max_revisions"]:
                print("---DECISION: MAX REVISIONS REACHED, STOPPING---")
                return "stop"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
    else:
        if state["first_attempt"] == True:
            return "not useful"
        
        if state["revision_number"] >= state["max_revisions"]:
            print("---DECISION: MAX REVISIONS REACHED, STOPPING---")
            return "stop"
        else:
            global NOT_SUPPORTED_COUNTER
            NOT_SUPPORTED_COUNTER += 1
            if NOT_SUPPORTED_COUNTER >= 3:
                print("---DECISION: TOO MANY UNSUPPORTED GENERATIONS, STOPPING---")
                return "stop" #We changed it because as sosn as main roter routes a query to web_search and the retrieved content doesn't satisfy for 3 continious web search rather than going to financial agent it stops and generate irrelevant result. 
                # return "financial_agent"
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRYING---")
            return "not supported"


def finance_tool_agent(state):
    """
    Call finance agent tools.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---FINANCE AGENT---",state["question"])
    question = state["question"]
    documents = state["documents"]
    global tokens
    global inp_tok
    
    # Invoke the finance agent
    with get_openai_callback() as cb:
        generation = finance_agent.invoke(question)
        web_search_flag = False
        for gen in generation:
            # print("GEN",gen)
            try:
                if "Will be right back" in gen['output'][0][0]:
                  web_search_flag = True
            except Exception as e:
                pass
            try:
                if "Failed to get" in gen['output']:
                  web_search_flag = True
            except Exception as e:
                pass
        # print("GENERATION",generation)
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    print(cb.total_tokens)
    print(cb.prompt_tokens)
    
    global WEB_SEARCH_COUNTER
    if web_search_flag and WEB_SEARCH_COUNTER == 0:
        generation.append({"output": web_search(state)["documents"][-1].page_content})
    
    final_generation = [item['output'] for item in generation]

    if len(final_generation)==0:
        final_generation = [web_search(state)["documents"][-1].page_content]
        
    final_generation_string=f"{final_generation}"
    
    return {
        "documents": documents,
        "question": question,
        "generation": final_generation_string,
        "revision_number": state["revision_number"], 
        "max_revisions": state["max_revisions"],
    }


def chart_Agents(generation):
    print("---CHART CREATING AGENT---")

    # Generate the chart reasoning response
    chart_data_response = chart_generator.invoke({"generation": generation})
    print(f"Chart Data Response: {chart_data_response}")

    # Check if the response indicates no chart is possible
    if "No chart possible" in chart_data_response:
        print("No chart can be generated from this context.")
        return "No chart created."

    try:
        # Extract the JSON from the response using regex
        json_match = re.search(r"\{.*\}", chart_data_response, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in the response.")

        json_text = json_match.group(0)  # Extract the JSON string
        chart_data = json.loads(json_text)  # Parse the JSON string

        # Determine chart type and delegate to the appropriate function
        chart_type = chart_data.get("chartType", "").lower()

        # Generate a unique file path
        save_dir = "charts"
        os.makedirs(save_dir, exist_ok=True)

        # Check if file already exists and create a unique name
        base_filename = "chart.png"
        save_path = os.path.join(save_dir, base_filename)
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(save_dir, f"chart_{counter}.png")
            counter += 1

        if chart_type == "bar":
            return generate_bar_chart(chart_data, save_path)
        elif chart_type == "line":
            return generate_line_chart(chart_data, save_path)
        elif chart_type == "pie":
            return generate_pie_chart(chart_data, save_path)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    except json.JSONDecodeError as jde:
        print(f"JSON parsing error: {jde}")
        return "Failed to create chart due to JSON parsing error."
    except Exception as e:
        print(f"Error in chart generation: {e}")
        return "Failed to create chart."


def reasoning_agent(state):
    """
    Reason based on the question and documents.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---REASON AGENT---")
    print("Question",state["question"])
    question = state["question"]
    documents = state["documents"]
    finance_agent_generation = state["generation"]
    # docs_content = [doc.page_content for doc in documents]
    global tokens, inp_tok

    with get_openai_callback() as cb:
        reasoning_output = reasoner.invoke({"question": question, "documents": finance_agent_generation})

        new_document = Document(page_content=reasoning_output, metadata={"source":"web search or finance tools"})
        state['documents'].append(new_document)

        if any(keyword in reasoning_output.lower() for keyword in ["chart", "graph", "visualize", "plot"]):
            print("Reasoning indicates a chart may need to be created. Trying to create the chart\n")
            save_path = chart_Agents(reasoning_output)
            if save_path is not None:
                print(f"Saved the image in: {save_path}")

    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    print(cb.total_tokens)
    print(cb.prompt_tokens)

    return {
        "documents": state['documents'],
        "question": question,
        "revision_number": state["revision_number"],
        "max_revisions": state["max_revisions"],
    }


def sql_agents(state):
    """
    Handle SQL-based questions by processing metadata and querying the database.

    Args:
        state (dict): The current graph state.

    Returns:
        state (dict): Updated state with the SQL query result.
    """
    print("---SQL QUERY---")
    print("Question",state["question"])
    # Process metadata and populate the database
    chunk = state["document"]
    table_maker = TableMaker(db_name="testDB.db")
    table_maker.process_chunk(chunk)

    # Initialize the SQLAgent
    sql_agent = SQLAgent(
        api_key=os.getenv["GROQ_API_KEY"],
        db_path="example.db",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Ask the SQLAgent for the answer
    question = state["question"]
    answer = sql_agent.ask_question(question)
    answer_string=f"{answer}"
    # print(f"SQL Query Result: {answer}")
    state["documents"].append(Document(page_content=answer_string), metadata={"source":"sql_agent_output"})
    state["first_attempt"] = False
    return state


workflow = StateGraph(GraphState)

# Define the nodes (same as provided)
workflow.add_node("web_search", web_search)  
workflow.add_node("retrieve", retrieve) 
workflow.add_node("grade_documents", grade_documents)  
workflow.add_node("generate", generate) 
workflow.add_node("transform_query", transform_query)
workflow.add_node("finance_agent", finance_tool_agent) 
workflow.add_node("reasoning_agent", reasoning_agent) 
workflow.add_node("sql_agents", sql_agents)

# Directly add an edge from START to the "retrieve" node
workflow.add_edge(START, "retrieve")

# workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents") #Based on web search it oftent directly generates output without financial tools
workflow.add_edge("web_search", "finance_agent")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "not_answerable": "finance_agent",
        "generate": "generate",
        "sql_agents" : "sql_agents",
    },
)

workflow.add_edge("finance_agent", "reasoning_agent")
workflow.add_edge("reasoning_agent", "generate")
workflow.add_conditional_edges(
    "transform_query",
    decide_to_retrieve,
    {
        "retrieve": "retrieve",
        "web_search": "web_search",
    },
)

workflow.add_conditional_edges(
    "sql_agents",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "stop": END,
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "stop": END,
    },
)



# Compile
app = workflow.compile()

from PIL import Image as PILImage
import io
import asyncio
import os 

output_dir =  os.path.join(os.getcwd()  ,"Pathway_chatbot")
os.makedirs(output_dir, exist_ok=True)

try:
    image_data = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    image = PILImage.open(io.BytesIO(image_data))
    image.save("workflow.png")

    # Write the mermaid code to a txt file
    mermaid_code = app.get_graph().draw_mermaid()
    with open("workflow_mermaid.txt", "w") as file:
        file.write(mermaid_code)
except Exception as e:
    print(f"An error occurred: {e}")


# Streamlit Interface
st.title("Query Processing Interface")
st.sidebar.title("Logs")



query_input = st.text_area("Enter your query", value="You can ask a question here.")
run_button = st.button("Run")
org_input = st.text_area("Organization", value="Only enter the name of the organization")
generate_FDD = st.button("Generate FDD Report")


async def Generator(subQueries, verification=" "):
    
    generations = []  # To store all generations

    async def process_subquery(sub_query):
        st.write(f"#### Processing {verification} query: {sub_query}")
        st.write("-" * 50)

        count = 0
        # Run the streaming process for each query
        async for output in app.astream({"question": sub_query, "revision_number": 0, "max_revisions": 2, "first_attempt":True}):
            # Capture output generation steps
            for key, value in output.items():
                st.write(f"#### At Node {key}")

            logs = logger_stream.get_logs()
            if logs:
                timestamp = int(time.time())
                unique_id = uuid.uuid4()
                st.sidebar.text_area("Logs", logs, height=200, max_chars=None, key=f"{sub_query}_{timestamp}_{unique_id}")

            count += 1
        return {"query": sub_query, "generation": value["final_generation"]}

    tasks = [process_subquery(sub_query) for sub_query in subQueries]
    generations = await asyncio.gather(*tasks)

    return generations

global data

global data

data = {
    "question_by_key": """ """,
    "question_by_business": """ """,
    "question_by_executive": """ """,
    "answer_for_key": """ """,
    "answer_for_business" : """ """,
    "answer_for_executive" : """ """,
    "agent_used" : """""",
    "suggestion_by_key" : """ """,
    "suggestion_by_business" : """ """,
    "suggestion_by_executive": """ """,
    "Is_key_metrics_satisfied" : "1",
    "Is_business_metrics_satisfied" : "1",
    "Is_executive_metrics_satisfied" : "1"
}

filename = "text_output.txt"
filename2 = "gen_output.txt"
global FDD_rev

FDD_rev = 0

Rev_limit = 2
from tools.helper import append_to_file
global final_report
final_report = """"""

def getMajorityVote(data):
    keys_to_check = [
        "Is_key_metrics_satisfied",
        "Is_business_metrics_satisfied",
        "Is_executive_metrics_satisfied"
    ]

    unsatisfied_count = sum(1 for key in keys_to_check if data.get(key) == "0")

    return unsatisfied_count


def Discussion(company,data,key_agent,executive_agent,business_agent):

    global FDD_rev
    prompt = f"There are questions asked by each agent and their respective answers in {data}. Discuss among other agents and suggest changes if any"
    # st.write(data)        
    
    key_suggestion,agent_used = key_agent(mode="Disc")
    data["suggestion_by_key"] = key_suggestion.invoke(prompt)

    first_line = data["suggestion_by_key"].splitlines()[0].strip()
    # st.write(first_line)
    if "Not Satisfied" in first_line:
        data["Is_key_metrics_satisfied"] = "0"
    elif "Satisfied" in first_line:
        data["Is_key_metrics_satisfied"] = "1"
    business_suggestion,agent_used = business_agent(mode="Disc")
    data["suggestion_by_business"] = business_suggestion.invoke(prompt)

    first_line = data["suggestion_by_business"].splitlines()[0].strip()
    # st.write(first_line)
    if "Not Satisfid" in first_line:
        data["Is_business_metrics_satisfied"] = "0"
    elif "Satisfied" in first_line:
        data["Is_business_metrics_satisfied"] = "1"

    executive_suggestion,agent_used = executive_agent(mode="Disc")
    data["suggestion_by_executive"] = executive_suggestion.invoke(prompt)

    first_line = data["suggestion_by_executive"].splitlines()[0].strip()
    # st.write(first_line)
    if "Not Satisfid" in first_line:
        data["Is_executive_metrics_satisfied"] = "0"
    elif "Satisfied" in first_line:
        data["Is_executive_metrics_satisfied"] = "1"
        
    # Last check to make sure that the final answers are appended
    if FDD_rev >= Rev_limit:
        final_txt = data["answer_for_key"] + "\n" + data["answer_for_business"] + "\n" + data["answer_for_executive"]
        append_to_file(filename,final_txt)
        return

    if getMajorityVote(data) > 1:
        st.write("### More than 2 agents are Not satsified! Running Reframer")

        # st.write(data)

        pmt1 = f"""Look at these questions asked by key_metrics_agent in {data["question_by_key"]}, and suggestion by other agents : {data["suggestion_by_key"]}, {data["suggestion_by_business"]}, {data["suggestion_by_executive"]},remake relevant questions for this agent."""
        key_queries_new = Disc_question_reframer.invoke(pmt1)

        pmt2 = f"""Look at these questions asked by business_agent in {data["question_by_business"]}, and suggestion by other agents : {data["suggestion_by_key"]}, {data["suggestion_by_business"]}, {data["suggestion_by_executive"]},remake relevant questions for this agent."""
        business_queries_new = Disc_question_reframer.invoke(pmt2)

        pmt3 = f"""Look at these questions asked by executive_agent in {data["question_by_executive"]}, and suggestion by other agents : {data["suggestion_by_key"]}, {data["suggestion_by_business"]}, {data["suggestion_by_executive"]},remake relevant questions for this agent."""
        executive_queries_new = Disc_question_reframer.invoke(pmt3)

        net_queries_new = [key_queries_new,business_queries_new,executive_queries_new]


        FDD_rev += 1
        
        st.write("### FDD_REV",FDD_rev)

        process_question(company, net_queries_new)



def process_question(company, netQueries):

    """
    Process a given question by decomposing it into sub-queries, running the workflow for each sub-query,
    and aggregating the answers to generate a final answer.

    Args:
        question (str): The input question to process.

    Returns:
        None: The function handles displaying results directly in the Streamlit interface.
    """

    if FDD_rev >= Rev_limit:
        st.write("### Rev Limit Hit")
        return

    global final_report

    for queries in netQueries:
        logger_stream.clear_logs()  # Clear any previous logs

        # Decompose the input query into sub-queries
        inputs = {"question": queries, "revision_number": 0, "max_revisions": 2}
        st.write("### original Questions: \n",inputs["question"])

        # Initialize the NOT_SUPPORTED_COUNTER
        global NOT_SUPPORTED_COUNTER, WEB_SEARCH_COUNTER
        NOT_SUPPORTED_COUNTER = 0
        WEB_SEARCH_COUNTER = 0

        with get_openai_callback() as cb:
            subQueries = decomposer.invoke(inputs).sub_queries

        global tokens, inp_tok
        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens
        
        if subQueries == []:
            subQueries.append(queries)

        st.write("### Subqueries:",subQueries)
        # Generate outputs for the sub-queries
        generations = asyncio.run(Generator(subQueries, verification=" "))
        answers = []
        for gen in generations:
            answers.append(gen["generation"])

        # Aggregate answers and display final answer
        if generations:
            with get_openai_callback() as cb:
                ans = aggregator.invoke({"question": queries, "answers": ", ".join(answers)})
                
            tokens += cb.total_tokens
            inp_tok += cb.prompt_tokens
            # reply = key_metrics_handler.invoke(ans.answer)
            # print(reply)
            st.write("### Final Answer:")
            st.write(ans.answer)

            st.write(tokens)
            st.write(inp_tok)

        for gen in generations:
            with st.expander(f"Sub-query: {gen['query']}"):
                st.write(gen["generation"])

            if queries == key_queries:
                with open("key_metrics_sub.txt", 'a') as file:

                    file.write("Question :" + gen["query"] + '\n')
                    file.write("Answer: " + gen["generation"] + '\n')  # Append the text followed by a newline
                    st.write("Answers to the key subqueries has been appended\n")

            with open(filename2, 'a') as file:
                file.write("Question :" + gen["query"] + '\n')
                file.write("Answer: " + gen["generation"] + '\n')  # Append the text followed by a newline

                st.write("Answers to the subqueries has been appended\n")

        with open(filename2,'r') as f:
            content = f.read()
            if queries == key_queries:
                save_to_pdf(content,"key_metric.pdf")

                key_generation = FDD_Generator_handeler.invoke(f"Agent is key_metrics and company is {company}" + content)
                final_report = final_report + key_generation + "\n"
            elif queries == business_quries:
                business_generation = FDD_Generator_handeler.invoke(f"Agent is business_agent and company is {company}" + content)
                final_report = final_report + business_generation + "\n"
            elif queries == exec_queries:
                executive_generation = FDD_Generator_handeler.invoke(f"Agent is executive agent and company is {company}" + content)

                final_report = final_report + executive_generation + "\n"

        if os.path.exists(filename2):
            os.remove(filename2)
            st.write(f"{filename2} has been deleted")


        if queries == key_queries:
            data["question_by_key"] = queries
            data["answer_for_key"] = ans.answer
        elif queries == business_quries:
            data["question_by_business"] = queries
            data["answer_for_business"] = ans.answer
        else:
            data["question_by_executive"] = queries
            data["answer_for_executive"] = ans.answer

    append_to_file("final_reportt.txt",final_report)

    # gen_rep = FDD_Generator_handeler.invoke(final_report)
    st.write("### FINAL REPORT_ENHANCED:\n",final_report)
    save_to_pdf(final_report,f"{org_input}_report_gen.pdf")
    Generated_json = get_json(final_report)
    # with open("generated_json.txt", 'a') as tmp_file:
    #     tmp_file.write(str(Generated_json))
    #     tmp_file.write("----------------------\n")
    st.write(Generated_json)
    # Save JSON to a file
    with open(f"report_dashboard.json", "w") as json_file:
        json.dump(Generated_json, json_file, indent=4)

    
    Discussion(company=company,data=data,key_agent=key_metrics_agent_mode,business_agent=business_model_agent_mode,executive_agent=executive_agent_mode)

    final_txt = data["answer_for_key"] + "\n" + data["answer_for_business"] + "\n" + data["answer_for_executive"]
    append_to_file(filename,final_txt)
    
    st.write(tokens)
    st.write(inp_tok)
    
    st.sidebar.write(logger_stream.get_logs())



from reportlab.lib.pagesizes import letter


from tools.helper import save_to_pdf
import uuid


if generate_FDD and org_input:
    
    # tmp = query_input
    # query_input = pii_remover(tmp)
    st.write(org_input)
    
    try:
        if llamaguard(org_input) != "safe":
            st.write(f"Query is unsafe: {llamaguard(org_input)}")
            st.stop()
    except Exception as e:
        st.write(f"LlamaGuard failed to check the query: {e}")
        print(f"LlamaGuard failed to check the query: {e}")
    
    # Check if the file exists and delete it
    MoveCharts()
    if os.path.exists(filename):
        os.remove(filename)
        print(f"{filename} already exists. It has been deleted.")


    input_queries101 = f"Ask the relevant Questions to the RAG about {org_input} to produce a relavant Financial Due Deligence Report."
    exec_queries = executive_agent_mode(mode="QnA").invoke(input_queries101)
    business_quries = business_model_agent_mode(mode="QnA").invoke(input_queries101)
    key_queries = key_metrics_agent_mode(mode="QnA").invoke(input_queries101)

    net_queries = [key_queries,exec_queries,business_quries]


    process_question(org_input,net_queries)
    final_txt = data["answer_for_key"] + "\n" + data["answer_for_business"] + "\n" + data["answer_for_executive"]
    append_to_file(filename,final_txt)
    


if run_button and query_input:
    logger_stream.clear_logs()  # Clear any previous logs
    
    # tmp = query_input
    # query_input = pii_remover(tmp)
    st.write(query_input)
    
    try:
        if llamaguard(query_input) != "safe":
            st.write(f"Query is unsafe: {llamaguard(query_input)}")
            st.stop()
    except Exception as e:
        st.write(f"LlamaGuard failed to check the query: {e}")
        print(f"LlamaGuard failed to check the query: {e}")

    # Decompose the input query into sub-queries
    inputs = {"question": query_input, "revision_number": 0, "max_revisions": 2}
    
    # Initialize the NOT_SUPPORTED_COUNTER
    NOT_SUPPORTED_COUNTER = 0
    WEB_SEARCH_COUNTER = 0

    with get_openai_callback() as cb:
        subQueries = decomposer.invoke(inputs).sub_queries

    # Track token usage
    tokens += cb.total_tokens
    inp_tok += cb.prompt_tokens
    
    if subQueries == []:
        subQueries.append(query_input)
    # subQueries = [inputs["question"]]

    st.write("### Sub-queries:", subQueries)

    # Generate outputs for the sub-queries
    generations = asyncio.run(Generator(subQueries, verification=" "))
    answers = []
    for gen in generations:
        answers.append(gen["generation"])

    # Aggregate answers and display final answer
    if generations:
        with get_openai_callback() as cb:
            ans = aggregator.invoke({"question": query_input, "answers": ", ".join(answers)})


        tokens += cb.total_tokens
        inp_tok += cb.prompt_tokens
        st.write("### Final Answer:")
        st.write(ans.answer)
        st.write(tokens)
        st.write(inp_tok)

    # Display each subquery result in a dropdown
    for gen in generations:
        with st.expander(f"Sub-query: {gen['query']}"):
            st.write(gen["generation"])

    # Show final log state
    
    st.sidebar.write(logger_stream.get_logs())