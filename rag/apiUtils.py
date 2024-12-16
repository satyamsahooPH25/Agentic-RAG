import requests
import time
import logging


def unstructured_request(
    api_key: str,
    contents: bytes,
    strategy: str = "hi_res",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> requests.Response | None:
    """
    Sends a request to the Unstructured API with retry logic.

    Args:
        - api_key: The API key for authentication.
        - contents: The document contents in bytes.
        - strategy: The parsing strategy (default: "hi_res").
        - max_retries: Maximum number of retry attempts (default: 3).
        - retry_delay: Delay between retries in seconds (default: 2.0).

    Returns:
        The response object if successful; returns None if all retries fail.
    """
    api_url = "https://api.unstructuredapp.io/general/v0/general"
    headers = {
        "accept": "application/json",
        "unstructured-api-key": api_key,
    }
    files = {"files": contents}
    data = {"strategy": strategy}

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, files=files, data=data)
            response.raise_for_status()  # Ensure request was successful
            return response

        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Returning None.")
                return None  # Signal failure
