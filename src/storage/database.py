import sqlite3
from typing import Dict, Any, List, Optional
from src.storage.table import Table
import os


class DatabaseManager:
    def __init__(self, db_name: str = "agent_data.db", reset_db: bool = False):
        """
        Initialize the database connection. If reset_db is True, creates a new database on each run.

        :param db_name: The name of the SQLite database file.
        :param reset_db: Whether to reset (delete) the database file on initialization.
        """
        self.db_name = db_name
        if reset_db and os.path.exists(db_name):
            os.remove(db_name)
        self.connection = sqlite3.connect(db_name)
        self.tables: Dict[str, Table] = {}
        self._create_default_tables()

    def _create_default_tables(self) -> None:
        """
        Create default tables for storing episode history using the Table class.
        """
        self.tables['episodes'] = Table(
            self.connection,
            'episodes',
            {
                'environment_name': 'TEXT',  # Clearer name for the simulation ID
                'simulation_id': 'INTEGER',  # Clearer name for the simulation ID
                'episode_number': 'INTEGER',  # Indicates the specific episode number
                'agent_id': 'INTEGER',  # ID of the agent involved in the episode
                'role': 'TEXT NOT NULL',  # Role of the speaker ('user', 'assistant', etc.)
                'content': 'TEXT NOT NULL',  # Content of the message or action
                'action': 'TEXT',  # Optional: Specific action taken by the agent
                'timestamp': 'DATETIME DEFAULT CURRENT_TIMESTAMP'  # Auto-captures the timestamp
            }
        )

    def create_table(self, table_name: str, columns: Dict[str, str]) -> None:
        """
        Dynamically creates a new table.

        :param table_name: Name of the new table.
        :param columns: Dictionary of column names and their data types.
        """
        self.tables[table_name] = Table(self.connection, table_name, columns)

    def __getitem__(self, table_name: str) -> Optional[Table]:
        """
        Allows dictionary-like access to tables within the manager.

        :param table_name: Name of the table to access.
        :return: The Table object if it exists, else None.
        """
        return self.tables.get(table_name)

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.connection.close()
