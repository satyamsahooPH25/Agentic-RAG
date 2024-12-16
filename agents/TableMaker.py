import sqlite3
import pandas as pd
from bs4 import BeautifulSoup
import json


class TableMaker:
    def __init__(self, db_name="example.db"):
        self.conn = sqlite3.connect(db_name)

    def generate_tables(self, html_content):
        """Converts all HTML tables in the content into DataFrames."""
        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")
        tables = soup.find_all("table")
        dataframes = []

        for index, table in enumerate(tables):
            # Extract headers (assumes up to 2 header rows)
            headers = []
            for header_row in table.find_all("tr")[:2]:
                headers += [header.text.strip() for header in header_row.find_all("th")]

            # Remove any empty headers due to colspan or rowspan
            headers = [header for header in headers if header]

            # Extract table rows
            rows = []
            for row in table.find_all("tr")[2:]:  # Skipping the first two header rows
                cells = [
                    cell.text.strip().replace("\n", " ").replace("$", "").strip()
                    for cell in row.find_all("td")
                ]
                rows.append(cells)

            # Convert to DataFrame if headers and rows exist
            if headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                dataframes.append((df, f"table_{index + 1}"))

        return dataframes

    def push_to_database(self, df, table_name):
        """Pushes the DataFrame to the SQLite database."""
        df.to_sql(name=table_name, con=self.conn, if_exists="replace", index=False)
        print(f"Table '{table_name}' pushed to the database.")

    def process_chunk(self, chunk):
        """Processes the chunk and extracts tables from the HTML."""
        html_content = chunk["metadata"]["text_as_html"]
        dataframes = self.generate_tables(html_content)

        for df, table_name in dataframes:
            self.push_to_database(df, table_name)