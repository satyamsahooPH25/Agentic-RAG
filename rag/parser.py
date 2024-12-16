from __future__ import annotations

import logging

import pathway as pw
from apiUtils import unstructured_request

logger = logging.getLogger(__name__)


class Parsers:

    def __init__(self, api_key: str):

        self.api_key = api_key

    def chipper(self, contents: bytes) -> list[tuple[str, dict]]:
        """
        Using the 'Chipper' model to parse the given document:

        Args:
            - contents: document contents

        Returns:
            a list of pairs: text chunk and metadata
            The metadata is obtained from Unstructured, you can check possible values
            Implements the Chipper method of Parsing and Chunking
        """

        response = unstructured_request(self.api_key, contents)
        # Fallback if the main request fails
        if response is None:
            logging.warning("Parser Failed !")
            return ConnectionError

        # print(response)
        
        elements = response.text
        # print(elements)
        docs = []
        if isinstance(elements, list):
            for element in elements:
                docs.append((element["text"], element["metadata"]))
        else:
            convert = eval(elements)
            for element in convert:
                # Add element["type"] and element["element_id"] to metadata
                element["metadata"]["type"] = element["type"]
                element["metadata"]["element_id"] = element["element_id"]
                docs.append((element["text"], element["metadata"]))

        # Open a file and write the content of docs in it
        with open("parsed_docs.txt", "w", encoding="utf-8") as file:
            for text, metadata in docs:
                file.write(f"Text: {text}\n")
                file.write(f"Metadata: {metadata}\n\n")
        file.close()
        print("Parsing Completed Successfully!")

        return docs

    def __call__(self, contents: pw.ColumnExpression) -> list[tuple[str, dict]]:
        """
        Parse the given document.

        Args:
            - contents: document contents

        Returns:
            a list of pairs: text chunk and metadata
            The metadata is obtained from Unstructured, you can check possible values
        """

        parsed_output = self.chipper(contents)
        return parsed_output
