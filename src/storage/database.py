import sqlite3
import os
import json

class DatabaseManager:
    def __init__(self, db_name="agent_data.db", reset_db=False):
        """
        Initialize the database connection. If reset_db is True, creates a new database on each run.
        """
        self.db_name = db_name
        if reset_db and os.path.exists(db_name):
            os.remove(db_name)
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self._create_tables()

    def _create_tables(self):
        """
        Create tables for storing episode history.
        """
        table = """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id INTEGER,
            agent_id INTEGER,
            history TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.cursor.execute(table)
        self.connection.commit()

    def insert_episode_history(self, episode_id, agent_id, history):
        """
        Insert the entire history of actions and observations as a single JSON string.
        """
        history_json = json.dumps(history)
        query = "INSERT INTO episodes (episode_id, agent_id, history) VALUES (?, ?, ?)"
        self.cursor.execute(query, (episode_id, agent_id, history_json))
        self.connection.commit()

    def close(self):
        """
        Close the database connection.
        """
        self.connection.close()
