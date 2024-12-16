def extract_relevant_fields(documents):
    parsed_documents = []

    for doc in documents:
        if doc.metadata is None:
            relevant_data = {
                "content": doc.page_content
            }
            # parsed_documents.append(relevant_data)
        metadata = doc.metadata

        # Extract relevant fields from metadata with checks for presence
        relevant_data = {
            "filename": metadata.get("filename", None),
            "filetype": metadata.get("filetype", None),
            "id": metadata.get("id", None),
            "name": metadata.get("name", None),
            "modified_time": metadata.get("modifiedTime", None),
            "page_number": metadata.get("page_number", None),
            "url": metadata.get("url", None),
            "content": doc.page_content,  # Main content for LLM generation
            "text_as_html": metadata.get(
                "text_as_html", None
            ),  # Include text_as_html if present
        }

        # Append only if at least one relevant field is present (optional)
        if any(value is not None for value in relevant_data.values()):
            parsed_documents.append(relevant_data)

    return parsed_documents


def format_document_fields(documents):
    # Create a string representation of the extracted fields
    formatted_strings = []

    for doc in documents:
        fields = []
        for key, value in doc.items():
            if value is not None:  # Only include fields that are not None
                fields.append(f"{key}: {value}")
        formatted_strings.append("\n".join(fields))  # Join fields for each document

    return "\n\n".join(formatted_strings)  # Separate documents by two newlines
