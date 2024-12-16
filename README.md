# FA3STER: Financial Agentic Autonomous and Accurate System Through Evolving Retrieval-Augmented Generation

**Pathway End Term Submission**

## Overview

FA3STER is an innovative multi-agent Retrieval-Augmented Generation (RAG) system designed to streamline the financial due diligence (FDD) process. By leveraging advanced dynamic indexing, contextual chunking, and autonomous agent collaboration, FA3STER transforms FDD into a fast, efficient, and accurate task.

---

## Architecture

The FA3STER system is built around a robust four-component architecture:

### **1. Pre-Retrieval and Retrieval Phase**
- **Agentic Chunker**: Enhances document parsing with GPT-4o-mini for context-aware chunking, ensuring precise retrieval.
- **Dynamic Data Management**: Utilizes Pathway's Real-Time Streaming Framework with Google Drive connector for real-time updates and ingestion.
- **Hybrid Indexing**: Combines semantic vector embeddings (OpenAI’s text-embedding-3-small) with metadata filtering for efficient data retrieval.
- **Unstructured Parser**: Extracts structured data like tables and key-value pairs for better context in financial documents.
- **Cohere Re-ranker**: Scores documents based on relevance to prioritize critical information.

### **2. Post-Retrieval Phase**
- **LangGraph-Based Workflow**: Enables seamless communication between agents using a SELF-RAG architecture.
- **Agentic Framework**:
  - **SQL Agent**: Extracts precise data from tables.
  - **Finance Agent**: Handles complex financial queries like stock trends, financial metrics, and earnings data.
  - **Reasoning Agent**: Generates actionable insights and visualizations like statistical graphs.
  - **Web Search Integration**: Fetches real-time external data via Tavily Search.

### **3. Vertical Autonomous Layer**
- A novel approach that vertically scales the system by interconnecting agents:
  - **Key Metrics Agent**: Focuses on financial health indicators like revenue and growth rates.
  - **Business Agent**: Evaluates market conditions and risks.
  - **Executive Agent**: Assesses governance, performance, and alignment with industry standards.
- **Collaborative Workflow**: Agents engage in iterative Q&A and discussions to refine results until a high-quality report is generated.

### **4. User Interface**
- **Chat Mode**: Interactive Q&A with visualized query processing.
- **Report Generation Mode**: Generates concise FDD reports and an actionable dashboard with key metrics and insights.

---

## Key Features

- **Transparency**: Real-time visualization of agent workflows and decision-making paths.
- **Error Resilience**: Robust error handling with fallback mechanisms.
- **Efficiency**: Processes queries in under 3 minutes with asynchronous and parallel operations.
- **Scalability**: Dynamic indexing and modular architecture ensure scalability for large datasets.

---

## Results and Metrics

### **Evaluation Datasets**
- **FinQABench**: Assessed performance on financial queries with 100 test cases.
- **SEC-10Q Subset**: Tested robustness on complex multi-document queries.

### **Performance Metrics**
| Workflow                          | Context Precision | Context Recall | Response Relevancy | Faithfulness | Factual Correctness |
|-----------------------------------|-------------------|----------------|---------------------|--------------|---------------------|
| Naive RAG w/ Agentic Chunking     | 0.904             | 0.849          | 0.928              | 0.901        | 0.518               |
| Post-Retrieval Agentic Workflow   | 0.690             | 0.396          | 0.701              | 0.788        | 0.384               |

---

## Challenges and Solutions

1. **Transparency**: Introduced socket-based communication for real-time visualization of query workflows.
2. **Parallelization**: Used `asyncio` for asynchronous query processing.
3. **Error Handling**: Implemented fallback mechanisms and logging for robust error management.

---

## Technology Highlights

- **Pathway Framework**: Core infrastructure for streaming, indexing, and processing.
- **OpenAI, Cohere, Unstructured.io**: Advanced tools for embeddings, ranking, and data parsing.
- **Llamaguard’s Safety Guardrails**: Ensures compliance with ethical standards.

---

## Conclusion

FA3STER exemplifies a transformative RAG application in the financial domain, leveraging innovative architecture and intelligent agents to revolutionize financial due diligence. By ensuring high accuracy, transparency, and efficiency, FA3STER delivers significant value to stakeholders, paving the way for intelligent automation in complex analytical tasks.

---

## Future Work

- **Enhanced Scalability**: Incorporating real-time market data through advanced ETL pipelines.
- **Improved Explainability**: Refining agent communication for clearer output justifications.



