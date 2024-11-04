import sqlite3
import json
from typing import Dict, Any, List, Optional


class Table:
    def __init__(self, connection: sqlite3.Connection, name: str, columns: Dict[str, str]):
        """
        Initializes the Table with a name, columns, and an active database connection.

        :param connection: The SQLite database connection.
        :param name: Name of the table.
        :param columns: A dictionary where keys are column names and values are SQLite data types.
        """
        self.connection = connection
        self.name = name
        self.columns = columns
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self) -> None:
        """Creates a table in the database based on the specified columns."""
        columns_def = ', '.join([f"{col_name} {col_type}" for col_name, col_type in self.columns.items()])
        create_query = f"CREATE TABLE IF NOT EXISTS {self.name} ({columns_def})"
        self.cursor.execute(create_query)
        self.connection.commit()

    def insert(self, **kwargs: Any) -> None:
        """
        Inserts a row into the table, automatically serializing dictionaries or lists to JSON.

        :param kwargs: Column-value pairs for the row to insert.
        """
        # Serialize dictionaries and lists to JSON strings automatically
        serialized_values = {
            key: (json.dumps(value) if isinstance(value, (dict, list)) else value)
            for key, value in kwargs.items()
        }

        columns = ', '.join(serialized_values.keys())
        placeholders = ', '.join(['?' for _ in serialized_values])
        insert_query = f"INSERT INTO {self.name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(insert_query, tuple(serialized_values.values()))
        self.connection.commit()

    def fetch_all(self) -> List[tuple]:
        """
        Fetches all rows from the table.

        :return: List of rows in the table.
        """
        select_query = f"SELECT * FROM {self.name}"
        self.cursor.execute(select_query)
        return self.cursor.fetchall()

    def fetch_by_column(self, column_name: str, value: Any) -> List[tuple]:
        """
        Fetches rows where a specific column matches a given value.

        :param column_name: The column to filter by.
        :param value: The value to match.
        :return: List of matching rows.
        """
        query = f"SELECT * FROM {self.name} WHERE {column_name} = ?"
        self.cursor.execute(query, (value,))
        return self.cursor.fetchall()


