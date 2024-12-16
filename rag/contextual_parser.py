from __future__ import annotations

import logging
import json
import re

import pathway as pw
from apiUtils import unstructured_request

from typing import List, Tuple, Dict
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import logging
from math import ceil


class ContextualProcessor:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
        
    def process_chunks(self, docs: List[Tuple[str, Dict]], document_chunk_size: int = 200, 
                      context_window_size: int = 20) -> List[Tuple[str, Dict]]:
        """
        Process document chunks to add contextual information using GPT-4.
        
        Args:
            docs: List of tuples containing (text, metadata)
            document_chunk_size: Number of chunks to consider as one document context
            context_window_size: Number of chunks to process at once with GPT-4
            
        Returns:
            Enhanced list of tuples with contextualized text and original metadata
        """
        # Group documents into larger chunks for context
        num_docs = len(docs)
        num_document_groups = ceil(num_docs / document_chunk_size)
        enhanced_docs = []
        
        for doc_group_idx in range(num_document_groups):
            # Get the current document group
            start_idx = doc_group_idx * document_chunk_size
            end_idx = min(start_idx + document_chunk_size, num_docs)
            current_doc_group = docs[start_idx:end_idx]
            
            # Create document context from all texts in the group
            document_context = "\n".join([text for text, _ in current_doc_group])
            
            # Process smaller windows within the document group
            num_windows = ceil(len(current_doc_group) / context_window_size)
            
            for window_idx in range(num_windows):
                window_start = window_idx * context_window_size
                window_end = min(window_start + context_window_size, len(current_doc_group))
                current_window = current_doc_group[window_start:window_end]
                
                # Extract just the texts for the current window
                window_texts = [text for text, _ in current_window]
                
                # Get contextual information from GPT-4
                contextualized_texts = self._get_contextual_information(
                    document_context=document_context,
                    window_texts=window_texts
                )
                
                # Combine enhanced texts with original metadata
                for (orig_text, metadata), enhanced_text in zip(current_window, contextualized_texts):
                    enhanced_docs.append((
                        f"Original_Content: {orig_text}\nContextualized_Context: {enhanced_text}",
                        metadata
                    ))
                
            logging.info(f"Processed document group {doc_group_idx + 1}/{num_document_groups}")
            
        return enhanced_docs
    
    def _get_contextual_information(self, document_context: str, window_texts: List[str]) -> List[str]:
        """
        Get contextual information for a window of texts using GPT-4o-mini.
        
        Args:
            document_context: The full context text for the current document group
            window_texts: List of text chunks to process
            
        Returns:
            List of contextual information for each text chunk
        """
        # Create the chunks text outside the main f-string
        chunks_text = '\n'.join(f'Chunk {i+1}: {text}' for i, text in enumerate(window_texts))

        # Now use it in the main f-string
        prompt = f"""Here is a document context:
        {document_context}

        I will provide you with several text chunks from this document. For each chunk, provide relevant contextual information that would help understand its place in the broader document. Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.

        Text chunks to analyze:
        {chunks_text}

        Provide context for each chunk in the following format:
        Chunk 1: <contextual information>
        Chunk 2: <contextual information>
        ...
        """
        
        try:
            response = self.llm([HumanMessage(content=prompt)])
            
            # Parse response to extract contextual information
            context_lines = response.content.strip().split('\n')
            contexts = []
            
            for line in context_lines:
                if line.startswith('Chunk '):
                    # Extract just the contextual information after the colon
                    context = line.split(':', 1)[1].strip()
                    contexts.append(context)
                    
            # Ensure we have a context for each input chunk
            if len(contexts) != len(window_texts):
                logging.warning(f"Mismatch in context count. Expected {len(window_texts)}, got {len(contexts)}")
                # Pad with empty contexts if necessary
                contexts.extend([''] * (len(window_texts) - len(contexts)))
                
            return contexts
            
        except Exception as e:
            logging.error(f"Error getting contextual information: {str(e)}")
            return [''] * len(window_texts)
        
logger = logging.getLogger(__name__)


class Parsers:

    def __init__(self, api_key: str, openai_key: str):
        self.api_key = api_key
        self.openai_key = openai_key
        self.call_count = 0  # Variable to track the number of times __call__ is invoked

    def chipper(self, contents: bytes) -> list[tuple[str, dict]]:
        """
        Using the 'Chipper' model to parse the given document.

        Args:
            - contents: document contents

        Returns:
            a list of tuples: text chunk and metadata
            The metadata is obtained from Unstructured, you can check possible values
            Implements the Chipper method of Parsing and Chunking
        """
        response = unstructured_request(api_key=self.api_key, contents=contents)  # Simulating the unstructured request
        if response is None:
            logging.warning("Parser Failed!")
            return []

        
        elements = response.json()
        print(elements)
        docs = []
        if isinstance(elements, list):
            for element in elements:
                print(element)
                # Add element["type"] and element["element_id"] to metadata
                if 'type' in element and 'element_id' in element:
                    element["metadata"]["type"] = element["type"]
                    element["metadata"]["element_id"] = element["element_id"]
                
                docs.append((element["text"], element["metadata"]))
        else:
            # Convert the string data back into the list of dictionaries (simulated here)
            convert = eval(elements)
            for element in convert:
                # Add element["type"] and element["element_id"] to metadata
                if 'type' in element and 'element_id' in element:
                    element["metadata"]["type"] = element["type"]
                    element["metadata"]["element_id"] = element["element_id"]
                docs.append((element["text"], element["metadata"]))

        print("Parsing Completed Successfully!")
        return docs

    def process_document_with_context(self, docs) -> list[tuple[str, dict]]:
        
        # Then process the chunks with contextual information
        processor = ContextualProcessor(api_key=self.openai_key)
        enhanced_docs = processor.process_chunks(docs)

        print("Context added to Doc!")
        return enhanced_docs

    def __call__(self, contents: bytes) -> list[tuple[str, dict]]:
        """
        Parse the given document.

        Args:
            - contents: document contents

        Returns:
            a list of pairs: text chunk and metadata
            The metadata is obtained from Unstructured, you can check possible values
        """
        self.call_count += 1  # Increment the call count each time __call__ is invoked
        print(self.call_count)
        parsed_output = self.chipper(contents)
        enhanced_docs = self.process_document_with_context(parsed_output)
        # print(enhanced_docs)
        return enhanced_docs

    def mock_unstructured_request(self, contents: bytes) -> list[tuple[str, dict]]:
        """
        A mock version of the unstructured request for simulation. 
        In reality, this would call some external API.
        
        Args:
            contents: document contents (bytes)
        
        Returns:
            A mock response containing a list of text chunks and metadata
        """
        return [
            ("This is the first chunk of text.", {"type": "paragraph", "element_id": 1}),
            ("This is the second chunk of text.", {"type": "paragraph", "element_id": 2})
        ]
