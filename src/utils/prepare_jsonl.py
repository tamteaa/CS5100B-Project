"""
This class handles the conversion of the SQLlite tables to JSON-L formats.
This JSON-L file can be used for fine-tuning.
"""
import json
import sqlite3


class SQLLiteToJSONLConverter:
    def __init__(self, db_name: str, table_name: str):
        """
        Constructor for the class

        :param db_name (str)      : Name of the database.

        :param table_name (str)   : Name of the table.
        """
        self.db_name = db_name
        self.table_name = table_name

    def write_conversational_jsonl(self, output_path: str, role_mapping: dict):
        """
        Converts the table to JSON-L format suitable for conversational fine-tuning specified by together.ai.
        For more information, see: https://docs.together.ai/docs/fine-tuning-data-preparation.

        :param output_path      :   Output path for the conversational JSON-L file.

        :param role_mapping     :   A mapping of SQL-lite table columns to 'role' and 'content'

        :return                 :   True if file is created, else Exception.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM {self.table_name};")
            columns = [desc[0] for desc in cursor.description]

            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))
                    json_line = {
                        "messages": [
                            {
                                "role": row_dict[role_mapping["role"]],
                                "content": row_dict[role_mapping["content"]]
                            }
                        ]
                    }
                    jsonl_file.write(json.dumps(json_line) + "\n")
            conn.close()
            return True

        except Exception as e:
            return str(e)


    def write_instructional_jsonl(self, output_path: str, prompt: str, completion: str):
        """
        Converts the table to JSON-L format suitable for instruction based fine-tuning specified by together.ai.
        For more information, see: https://docs.together.ai/docs/fine-tuning-data-preparation.

        :param output_path  : Output path for the conversational JSON-L file.

        :param prompt       :  Name of the column to be used as prompt.

        :param completion   : Name of the column to be used as completion.

        :return             : True if file is created, else Exception.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM {self.table_name};")
            columns = [desc[0] for desc in cursor.description]

            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))
                    json_line = {
                        "prompt": row_dict[prompt],
                        "completion": row_dict[completion]
                    }
                    jsonl_file.write(json.dumps(json_line) + "\n")

            conn.close()
            return True

        except Exception as e:
            return str(e)

