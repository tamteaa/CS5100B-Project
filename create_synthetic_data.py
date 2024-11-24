import os.path

from src.utils.prepare_jsonl import SQLLiteToJSONLConverter


class CreateSyntheticData:
    """
    Collects the data for training and fine-tuning.
    """
    def __init__(self, db_name:str, table_name: str ):
        """
        Constructor for this class

        :param db_name: DB name to collect the data for training and fine-tuning.

        :param table_name: table name to collect the data for training and fine-tuning. The data is converted to JSON-L
        """
        self.db_name = db_name
        self.table_name = table_name
        self.jsonl_converter = SQLLiteToJSONLConverter(self.db_name, self.table_name)
        self.jsonl_file_path = "fine_tuning.jsonl"

    def get_jsonl_file(self):
        """
        Creates the json-l file. The file will be available at the root directory.

        :return : None
        """
        jsonl_file_path = os.path.join(os.path.dirname(__file__), self.jsonl_file_path)
        status = self.jsonl_converter.write_instructional_jsonl(jsonl_file_path, prompt="", completion="")
        if status:
            print(f"JSON-L file for fine-tuning is available at {self.jsonl_file_path}")
        else:
            print(f"Encountered error: {status}")


