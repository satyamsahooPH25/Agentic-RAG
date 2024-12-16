This is the basic RAG implementation using Unstructured.io as parser, Pathway as vectorstore, Cohere Rerank3 for reranking, Google's text-embedding-004 for embedding and Gemini-1.5-Pro as LLM.

## Installation
1. Clone the repository
2. Install the requirements
3. Add the `credentials.json` file and `.env` file in the root directory
4. Run the ragServer.py file to run the pathway server
5. Run the rag.py file to run the client and get the result for your query

## .env file
```
UNSTRUCTURED_API_KEY = ""
COHERE_API_KEY = ""
GOOGLE_API_KEY = ""
```

0:"What is Apple's current operating margin?"
1:"What is the industry average operating margin in the technology sector?"
2:"Retrieve Apple's debt-to-equity ratio trends over the past five years."
3:"Identify any significant changes or outliers in Apple's debt-to-equity ratio trends over the past five years."
4:"What is Apple's current ratio?"
5:"What is the typical range for current ratios of large technology companies?"
6:"Analyze Apple's return on equity (ROE) over the past three years."
7:"Analyze Apple's return on assets (ROA) over the past three years."
8:"Compare Apple's ROE to industry benchmarks."
9:"Compare Apple's ROA to industry benchmarks."
10:"What is Apple's ESG score?"
11:"How does Apple's ESG score align with sustainability standards in the technology industry?"