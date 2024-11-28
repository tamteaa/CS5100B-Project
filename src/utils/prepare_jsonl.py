"""
This class handles the conversion of the SQLlite tables to JSON-L formats.
This JSON-L file can be used for fine-tuning.
"""
import json
import sqlite3
import os
import pandas as pd


class CreateSyntheticData:
    """
    Collects the data for training and fine-tuning, with support for both SQLite databases and DataFrames.
    """

    def __init__(self, db_name: str = None, table_name: str = None, df: pd.DataFrame = None):
        """
        Constructor for this class.

        :param db_name: DB name for collecting data for training and fine-tuning (optional).
        :param table_name: Table name for collecting data for training and fine-tuning (optional).
        :param df: DataFrame for direct conversion to JSON-L (optional).
        """
        self.db_name = db_name
        self.table_name = table_name
        self.df = df
        self.jsonl_file_path = "fine_tuning.jsonl"

    def validate_environments(self, training_envs: list, validation_envs: list, force: bool = False):
        """
        Validates that the specified training and validation environments exist in the data.
        Ensures there are no overlaps or mismatched environments.

        :param training_envs: List of environment names for training.
        :param validation_envs: List of environment names for validation.
        :param force: If True, skips missing environments validation.
        :raises ValueError: If any validation errors occur.
        """
        # Load environment names
        if self.df is not None:
            # Use the DataFrame to fetch environment names
            all_envs = set(self.df['environment_name'].unique())
        elif self.db_name and self.table_name:
            # Query the database to fetch environment names
            conn = sqlite3.connect(self.db_name)
            query = f"SELECT DISTINCT environment_name FROM {self.table_name}"
            all_envs = set(row[0] for row in conn.execute(query))
            conn.close()
        else:
            raise ValueError("Either a DataFrame or a database name and table name must be provided.")

        # Validate the environments
        specified_envs = set(training_envs + validation_envs)
        missing_envs = all_envs - specified_envs
        invalid_envs = specified_envs - all_envs
        overlap = set(training_envs) & set(validation_envs)

        errors = []
        if missing_envs and not force:
            errors.append(f"Environments not assigned to training or validation: {missing_envs}")
        if invalid_envs:
            errors.append(f"Invalid environments specified (not in the data): {invalid_envs}")
        if overlap:
            errors.append(f"Environments appear in both training and validation sets: {overlap}")

        if errors:
            raise ValueError("\n".join(errors))

        print("Environment validation passed.")

    def write_jsonl_from_df(self, output_path: str):
        """
        Converts the provided DataFrame into a JSON-L file.

        :param output_path: Path to save the JSON-L file.
        """
        if self.df is None:
            raise ValueError("No DataFrame provided for conversion.")
        converter = DataFrameToJSONLConverter(self.df)
        status = converter.write_conversational_jsonl(output_path)
        if status:
            print(f"JSON-L file created at {output_path}.")
        else:
            print(f"Error during JSON-L creation: {status}")

    def _validate_environments(self, training_envs: list, validation_envs: list, force: bool = False):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Get all environments from DB
        cursor.execute(f"SELECT DISTINCT environment_name FROM {self.table_name}")
        db_envs = set(row[0] for row in cursor.fetchall())

        specified_envs = set(training_envs + validation_envs)
        missing_envs = db_envs - specified_envs
        invalid_envs = specified_envs - db_envs
        overlap = set(training_envs) & set(validation_envs)

        errors = []
        if missing_envs and not force:
            errors.append(f"Environments not assigned: {missing_envs}")
        if invalid_envs:
            errors.append(f"Invalid environments specified: {invalid_envs}")
        if overlap:
            errors.append(f"Environments in both sets: {overlap}")

        conn.close()
        if errors:
            raise ValueError("\n".join(errors))

    def generate_split_data(self, training_envs: list, validation_envs: list, force: bool = False):
        """
        Splits the data into training and validation sets based on environment names and writes to JSON-L files.

        :param training_envs: List of environment names for training.
        :param validation_envs: List of environment names for validation.
        :param force: If True, allows skipping the validation for missing environments.
        :return: Paths to the generated training and validation JSON-L files.
        """
        # Validate environments
        self.validate_environments(training_envs, validation_envs, force)

        current_dir = os.getcwd()
        training_path = os.path.join(current_dir, "training_data.jsonl")
        validation_path = os.path.join(current_dir, "validation_data.jsonl")

        if self.df is not None:
            # Generate training and validation splits from the DataFrame
            training_df = self.df[self.df['environment_name'].isin(training_envs)]
            validation_df = self.df[self.df['environment_name'].isin(validation_envs)]

            # Convert to JSON-L
            print(f"Generating training data from environments: {training_envs}")
            DataFrameToJSONLConverter(training_df).write_conversational_jsonl(training_path)

            print(f"Generating validation data from environments: {validation_envs}")
            DataFrameToJSONLConverter(validation_df).write_conversational_jsonl(validation_path)

    def get_jsonl_file(self):
        """
        Creates the json-l file. The file will be available at the root directory.

        :return : None
        """

        current_dir = os.getcwd()
        jsonl_file_path = os.path.join(current_dir, self.jsonl_file_path)
        print(jsonl_file_path)
        status = self.jsonl_converter.write_conversational_jsonl(jsonl_file_path)
        if status:
            print(f"JSON-L file for fine-tuning is available at {jsonl_file_path}")
        else:
            print(f"Encountered error: {status}")

    def convert_to_llama_format(self, input_file, output_file):
        with open(output_file, "w") as out_f:
            with open(input_file, "r") as in_f:
                for line in in_f:
                    # Parse each line as a separate JSON object
                    data = json.loads(line.strip())

                    # Extract messages
                    messages = data["messages"]

                    # Convert to llama 3 format
                    output = "<|begin_of_text|>"
                    for message in messages:
                        role = message["role"]
                        content = message["content"]

                    if role == "system":
                        output += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                    elif role == "user":
                        output += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                    elif role == "assistant":
                        output += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

                    # Wrap the output in the required JSON format and write to file
                    final_output = json.dumps({"text": output})
                    out_f.write(final_output + "\n")


class SQLLiteToJSONLConverter:
    def __init__(self, db_name: str, table_name: str):
        """
        Constructor for the class

        :param db_name (str)      : Name of the database.

        :param table_name (str)   : Name of the table.
        """
        self.db_name = db_name
        self.table_name = table_name

    def write_conversational_jsonl(self, output_path: str, environment_names: list = None):
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            env_filter = ""
            if environment_names:
                placeholders = ','.join('?' * len(environment_names))
                env_filter = f"WHERE environment_name IN ({placeholders})"

            query = f"""
                SELECT simulation_id, agent_id, episode_number, role, content
                FROM {self.table_name}
                {env_filter}
                ORDER BY simulation_id, agent_id, timestamp;
            """

            cursor.execute(query, environment_names if environment_names else ())

            columns = [desc[0] for desc in cursor.description]

            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                current_conversation = []
                last_key = None

                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))

                    # Unique key for each conversation: combination of sim_id and agent_id
                    current_key = (row_dict["simulation_id"], row_dict["agent_id"])

                    # Detect conversation transitions (new sim_id or agent_id)
                    if last_key is not None and current_key != last_key:
                        # Write the previous conversation to the JSONL file
                        if current_conversation:
                            jsonl_file.write(json.dumps({"messages": current_conversation}) + "\n")
                        current_conversation = []

                    # Append message to the current conversation
                    current_conversation.append({
                        "role": row_dict["role"],
                        "content": row_dict["content"]
                    })

                    last_key = current_key

                # Write the final conversation if there's any data left
                if current_conversation:
                    jsonl_file.write(json.dumps({"messages": current_conversation}) + "\n")

                conn.close()
                return True

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
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


import json
import pandas as pd


class DataFrameToJSONLConverter:
    """
    A class for converting a DataFrame into a JSON-L file suitable for fine-tuning tasks.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Constructor for the DataFrame to JSON-L converter.

        :param df (pd.DataFrame): The DataFrame to convert to JSON-L format.
        """
        self.df = df

    def write_conversational_jsonl(self, output_path: str):
        """
        Converts the DataFrame into a JSON-L format for conversational fine-tuning.

        :param output_path: Path to save the JSON-L file.
        :return: True if the file is created successfully, otherwise raises an exception.
        """
        try:
            if not all(col in self.df.columns for col in ['simulation_id', 'agent_id', 'role', 'content']):
                raise ValueError("The DataFrame must contain 'simulation_id', 'agent_id', 'role', and 'content' columns.")

            # Sort the DataFrame by simulation_id, agent_id, and timestamp
            self.df = self.df.sort_values(by=['simulation_id', 'agent_id', 'timestamp'])

            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                current_conversation = []
                last_key = None

                for _, row in self.df.iterrows():
                    # Unique key for each conversation: combination of simulation_id and agent_id
                    current_key = (row["simulation_id"], row["agent_id"])

                    # Detect conversation transitions (new simulation_id or agent_id)
                    if last_key is not None and current_key != last_key:
                        # Write the previous conversation to the JSONL file
                        if current_conversation:
                            jsonl_file.write(json.dumps({"messages": current_conversation}) + "\n")
                        current_conversation = []

                    # Append the current message to the conversation
                    current_conversation.append({
                        "role": row["role"],
                        "content": row["content"]
                    })

                    last_key = current_key

                # Write the final conversation if there's any data left
                if current_conversation:
                    jsonl_file.write(json.dumps({"messages": current_conversation}) + "\n")

            return True

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            return str(e)

    def write_instructional_jsonl(self, output_path: str, prompt: str, completion: str):
        """
        Converts the DataFrame to JSON-L format for instruction-based fine-tuning.

        :param output_path: Path to save the JSON-L file.
        :param prompt: Column name to be used as the prompt.
        :param completion: Column name to be used as the completion.
        :return: True if the file is created successfully, otherwise raises an exception.
        """
        try:
            if prompt not in self.df.columns or completion not in self.df.columns:
                raise ValueError(f"Missing required columns '{prompt}' or '{completion}' in the DataFrame.")

            with open(output_path, "w", encoding="utf-8") as jsonl_file:
                for _, row in self.df.iterrows():
                    json_line = {
                        "prompt": row[prompt],
                        "completion": row[completion]
                    }
                    jsonl_file.write(json.dumps(json_line) + "\n")

            return True

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            return str(e)