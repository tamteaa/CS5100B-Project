"""
This class handles the conversion of the SQLlite tables to JSON-L formats.
This JSON-L file can be used for fine-tuning.
"""
import json
import sqlite3
import os


class CreateSyntheticData:
    """
    Collects the data for training and fine-tuning.
    """
    def __init__(self, db_name: str, table_name: str):
        """
        Constructor for this class

        :param db_name: DB name to collect the data for training and fine-tuning.

        :param table_name: table name to collect the data for training and fine-tuning. The data is converted to JSON-L
        """
        self.db_name = db_name
        self.table_name = table_name
        self.jsonl_converter = SQLLiteToJSONLConverter(self.db_name, self.table_name)
        self.jsonl_file_path = "fine_tuning.jsonl"

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
        self._validate_environments(training_envs, validation_envs, force)

        current_dir = os.getcwd()
        training_path = os.path.join(current_dir, "training_data.jsonl")
        validation_path = os.path.join(current_dir, "validation_data.jsonl")

        print(f"Generating training data from environments: {training_envs}")
        self.jsonl_converter.write_conversational_jsonl(training_path, training_envs)

        print(f"Generating validation data from environments: {validation_envs}")
        self.jsonl_converter.write_conversational_jsonl(validation_path, validation_envs)

        return training_path, validation_path

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

