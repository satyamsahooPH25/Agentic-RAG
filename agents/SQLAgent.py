import os
import re
import sqlite3

from google.generativeai import GenerativeModel, configure
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from google.generativeai import GenerativeModel, configure
from langchain_openai import ChatOpenAI

class SQLAgent:
    def __init__(self, api_key: str, db_path: str, google_api_key: str):
        # Set the required environment variable for the API key
        os.environ["GROQ_API_KEY"] = api_key

        # Configure the Google API key
        configure(api_key=google_api_key)

        # Create the SQLDatabase instance for a local SQLite database
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.db_path = db_path  # Store the database path for direct schema querying

        # Initialize the LLM
        # self.llm = ChatGroq(
        #     model="llama-3.2-90b-text-preview",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=45,
            max_retries=2,
        )

        # Create SQL query chain
        self.generate_query = create_sql_query_chain(self.llm, self.db)

        # Initialize the tool for executing the SQL query
        self.execute_query = QuerySQLDataBaseTool(db=self.db)

    def fetch_db_schema(self):
        """Fetches the schema of the database directly using SQLite queries."""
        schema_description = ""
        # Connect to the SQLite database directly to query schema information
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                # Get column names and types for each table
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                schema_description += (
                    f"Table {table} columns: "
                    + ", ".join([f"{col[1]} ({col[2]})" for col in columns])
                    + "\n"
                )

        return schema_description.strip()

    def ask_question(self, question: str):
        """This method generates and executes SQL queries, handling complex queries if necessary."""
        # Fetch the schema to help the LLM understand the tables and their structure
        schema = self.fetch_db_schema()

        # Enhanced prompt to guide the model for multi-table or JOIN queries if required
        prompt = f"""You are an SQL expert agent with access to a database.
        Here is the schema of the database:
        {schema}

        Given a question, you should generate an appropriate SQL query, even if it requires complex logic like JOINs.
        Question: '{question}'
        SQLQuery:"""

        # Generate SQL query for the given question
        query_result = self.generate_query.invoke({"question": prompt})
        query_match = re.search(r"SQLQuery:\s*(.*)", query_result)

        if not query_match:
            return "Failed to generate a valid SQL query."

        sql_query = query_match.group(1)

        # Execute the query on the database
        try:
            result = self.execute_query(sql_query)
        except Exception as e:
            return f"Execution error: {e}"

        # Use Google Generative AI to rephrase the answer
        model = GenerativeModel("gemini-pro")
        rephrased_response = model.generate_content(
            f"You are an SQL answering agent. There is a question: '{question}', I have a SQL query: '{sql_query}', and the result: '{result}'. Rephrase the answer in a proper manner."
        )

        return rephrased_response.text
