import os
from contextual_parser import Parsers

import pathway as pw
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm import parsers
from langchain_openai import OpenAIEmbeddings
from pathway.xpacks.llm import embedders

import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["UNSTRUCTURED_API_KEY"] = os.getenv("UNSTRUCTURED_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["GDRIVE_FOLDER_OBJECT_ID"] = os.getenv("GDRIVE_FOLDER_OBJECT_ID")

def initialize_data_sources():
    """Initialize data sources for Pathway VectorStore."""
    data_sources = [
        pw.io.gdrive.read(
            object_id=os.environ["GDRIVE_FOLDER_OBJECT_ID"],
            service_user_credentials_file="../credentials.json",
            with_metadata=True,
        )
    ]
    return data_sources


def setup_persistence_backend():
    """Set up persistence backend for VectorStoreServer."""
    persistence_backend = pw.persistence.Backend.filesystem("./vector_store/")
    return pw.persistence.Config(persistence_backend)


def create_embeddings_model():
    """Create and return the Google Generative AI embeddings model."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
    )


def create_text_splitter():
    """Create and return a character-based text splitter."""
    return CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)


def create_vector_store_server(data_sources, embeddings_model, parser):
    """
    Initialize and return the VectorStoreServer instance.

    Args:
        data_sources (list): A list of data sources to be used by the VectorStoreServer.
        embeddings_model (object): The embeddings model to be used for generating vector representations.
        parser (object): The parser to be used for processing the data sources.

    Returns:
        VectorStoreServer: An instance of VectorStoreServer initialized with the provided components.
    """
    return VectorStoreServer.from_langchain_components(
        *data_sources,
        embedder=embeddings_model,
        parser=parser,
    )



def run_vector_store_server(
    vector_server,
    persistence_backend,
    host="127.0.0.1",
    port=8011,
    threaded=False,
    with_cache=True,
):
    """
    Run the vector store server with the specified configuration.

    Parameters:
    vector_server (object): The vector server instance to be run.
    persistence_backend (object): The backend used for caching and persistence.
    host (str, optional): The hostname or IP address to bind the server to. Defaults to "127.0.0.1".
    port (int, optional): The port number to bind the server to. Defaults to 8011.
    threaded (bool, optional): Whether to run the server in threaded mode. Defaults to False.
    with_cache (bool, optional): Whether to enable caching. Defaults to True.

    Returns:
    None
    """
    vector_server.run_server(
        host=host,
        port=port,
        threaded=threaded,
        with_cache=with_cache,
        cache_backend=persistence_backend,
    )


def main():
    # Load environment variables
    load_environment_variables()

    # Initialize data sources, embeddings, text splitter, and persistence backend
    data_sources = initialize_data_sources()
    persistence_config = setup_persistence_backend()
    embeddings_model = create_embeddings_model()

    parser = Parsers(api_key=os.environ["UNSTRUCTURED_API_KEY"], openai_key=os.environ["OPENAI_API_KEY"])
    
    vector_server = create_vector_store_server(data_sources, embeddings_model, parser)
    print("Vector Store created...")
    
    # Run vector store server
    run_vector_store_server(vector_server, persistence_config.backend)
    print("Vector Store running...")


if __name__ == "__main__":
    main()
