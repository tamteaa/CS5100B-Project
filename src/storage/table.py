

class Table:
    def __init__(self, name, columns=None):
        """
        Initialize the table with a name and optional columns.
        Columns should be defined as a dictionary with column names as keys
        and data types as values, e.g., {'name': 'TEXT', 'age': 'INTEGER'}.
        """
        self.name = name
        self.columns = columns if columns is not None else {}
        self.data = []

    def create_table_sql(self):
        """Generate a SQL CREATE TABLE statement."""
        columns_def = ", ".join([f"{col} {dtype}" for col, dtype in self.columns.items()])
        return f"CREATE TABLE {self.name} ({columns_def});"

    def insert_row(self, row):
        """
        Insert a row of data.
        The row should be a dictionary mapping column names to values.
        """
        if not set(row.keys()).issubset(self.columns.keys()):
            raise ValueError("Row contains invalid column names.")
        self.data.append(row)

    def select_all(self):
        """Simulate a SQL SELECT * FROM table;"""
        return self.data

    def select_where(self, column, value):
        """Simulate a SQL SELECT * FROM table WHERE column = value;"""
        if column not in self.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        return [row for row in self.data if row.get(column) == value]

    def display(self):
        """Display the table in a tabular format, similar to SQL query results."""
        if not self.data:
            print(f"Table '{self.name}' is empty.")
            return

        # Print the column headers
        print(f"{' | '.join(self.columns.keys())}")
        print("-" * (len(self.columns) * 10))

        # Print the rows
        for row in self.data:
            print(" | ".join(str(row.get(col, '')) for col in self.columns.keys()))

