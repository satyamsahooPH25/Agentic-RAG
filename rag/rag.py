import os

from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import PathwayVectorClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


async def create_pathway_client(host="127.0.0.1", port=8011, k=10):
    """Initialize PathwayVectorClient and return as retriever.

    Args:
        host (str, optional): The host where the pathway is hosted. Defaults to "127.0.0.1".
        port (int, optional): The port which the pathway uses. Defaults to 8011.
        k (int, optional): The number of retrieval needed from the vectorstore. Defaults to 10.

    Returns:
        _type_: The retriever object.
    """
    client = PathwayVectorClient(host=host, port=port)
    return client.as_retriever(search_kwargs={"k": k})


def retrieve_relevant_documents(retriever, query):
    """Retrieve relevant documents based on the given query.

    Args:
        retriever (Retriever Object): The retriever object.
        query (str): The query to search for.

    Returns:
        _type_: The list of relevant documents.
    """
    try:
        with get_openai_callback() as cb:
            data_ret = retriever.invoke(query)
            print("hrere")
        return data_ret
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


def create_compressor(model="rerank-english-v3.0"):
    """Create and return a CohereRerank compressor."""
    return CohereRerank(model=model, top_n=5)


def compress_documents(retriever, query, compressor):
    """
    Compress relevant documents using the provided compressor.

    Args:
        retriever (BaseRetriever): The retriever to fetch relevant documents.
        query (str): The query string to search for relevant documents.
        compressor (BaseCompressor): The compressor to compress the retrieved documents.

    Returns:
        list: A list of compressed documents. If an error occurs, returns an empty list.

    Raises:
        Exception: If there is an error during document compression, it is caught and printed.
    """
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    try:
        with get_openai_callback() as cb:
            comp_doc = compression_retriever.invoke(query)
        return cb.total_tokens, cb.prompt_tokens, comp_doc
    except Exception as e:
        return 0, 0, retrieve_relevant_documents(retriever, query)
        # print(f"Error during document compression: {e}")
        # return 0, 0, []


def create_llm():
    """Create and return an instance of the ChatGoogleGenerativeAI LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def create_groq_llm():
    """Create and return an instance of the ChatGoogleGenerativeAI LLM."""
    return ChatGroq(
        model="llama-3.2-90b-vision-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )


def create_openai_llm():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=45,
        max_retries=2,
    )
    print("OpenAI LLM initialized.")
    return llm


def create_prompt_template():
    """Create and return a prompt template for the question-answering task."""
    return ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        Make sure to give the detailed location of the answer using the metadata provided. 
        For numerical answers, check whether the units are correct. Be sure to provide the answer in the correct format with correct units.
        For example, If the answer is from pdf named attention.pdf and the answer is from page 5, then in the answer mention the location as attention.pdf, page:5."
        I will tip you $1000 if the user finds the answer helpful. 
        <context>
        {context}
        </context>
        Question: {question}
        """
    )


def get_answer(context, question, llm, prompt):
    """
    Generate an answer based on the context and question using the LLM.

    Args:
        context (str): The context or passage from which the answer should be derived.
        question (str): The question that needs to be answered.
        llm (object): The language model instance used to generate the answer.
        prompt (object): The prompt or chain of prompts used to guide the language model.

    Returns:
        str: The generated answer based on the context and question.
        None: If an error occurs during the generation process.
    """
    # Warning: To be used only when using Chipper
    # extracted_fields = extract_relevant_fields(context)
    # formatted_context = format_document_fields(extracted_fields)
    # print(formatted_context)
    formatted_context = context
    rag_chain = prompt | llm | StrOutputParser()
    try:
        with get_openai_callback() as cb:
            ans_get = rag_chain.invoke(
                {"context": formatted_context, "question": question}
            )
        return cb.total_tokens, cb.prompt_tokens, ans_get

    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Could not generate answer: {e}"
        


# def main():
#     # Load environment variables
#     load_environment_variables()

#     # Initialize Pathway client and retrieve documents
#     retriever = create_pathway_client()
#     #query = "What is positional encoding in transformers?"
#     query = "Total Number of Class A Shares Purchased between 1 to 30 November"
#     # relevant_docs = retrieve_relevant_documents(retriever, query) # Docs without reranking

#     # Create compressor and compress documents
#     compressor = create_compressor()
#     compressed_docs = compress_documents(retriever, query, compressor)

#     # Create the LLM and prompt template
#     llm = create_llm()
#     prompt = create_prompt_template()

#     # Generate and print answer
#     answer = get_answer(compressed_docs, query, llm, prompt)
#     if answer:
#         print(answer)


# if __name__ == "__main__":
#     main()
