import os
import sqlite3


def merge_databases(input_dir, output_db):
    """
    Merges all SQLite .db files in a specified directory into a single database and counts the rows in each file.
    Supports databases with or without a 'score' column.

    Args:
        input_dir (str): Path to the directory containing input .db files.
        output_db (str): Path to the output .db file.
    """
    # Check if the output database already exists
    if os.path.exists(output_db):
        response = input(f"The output database '{output_db}' already exists. Do you want to delete it? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            os.remove(output_db)
            print(f"Deleted existing output database '{output_db}'.")
        else:
            print(f"Operation canceled. The output database '{output_db}' was not deleted.")
            return

    # Get all .db files in the input directory
    db_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.db')]

    if not db_files:
        print("No database files found in the specified directory!")
        return

    def count_rows(connection):
        """Count the total rows in all tables of a database."""
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        total_rows = 0
        for table_name, in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            rows = cursor.fetchone()[0]
            total_rows += rows
        return total_rows

    # Count rows in input databases
    for db_file in db_files:
        if not os.path.exists(db_file):
            print(f"Database file '{db_file}' not found. Skipping.")
            continue

        with sqlite3.connect(db_file) as conn:
            total_rows = count_rows(conn)
            print(f"Total rows in '{db_file}': {total_rows}")

    # Connect to the output database
    output_conn = sqlite3.connect(output_db)
    output_cursor = output_conn.cursor()

    for db_file in db_files:
        if not os.path.exists(db_file):
            continue

        # Connect to the current database file
        input_conn = sqlite3.connect(db_file)
        input_cursor = input_conn.cursor()

        # Get all table names from the input database
        input_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = input_cursor.fetchall()

        for table_name, in tables:
            # Get the schema of the table
            input_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = input_cursor.fetchall()
            column_names = [col[1] for col in columns]  # Extract column names

            # Ensure the 'score' column exists in the schema
            if 'score' not in column_names:
                print(f"Adding missing 'score' column to table '{table_name}' in '{db_file}'...")
                input_cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN score DOUBLE DEFAULT NULL")
                column_names.append('score')  # Add to column list

            # Ensure the output table schema matches the input table schema
            output_cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not output_cursor.fetchone():
                print(f"Creating table '{table_name}' in output database...")
                create_table_sql = f"CREATE TABLE {table_name} ({', '.join(f'{col[1]} {col[2]}' for col in columns)});"
                output_cursor.execute(create_table_sql)
                output_conn.commit()

            # Copy data from the input table to the output table
            input_cursor.execute(f"SELECT * FROM {table_name}")
            rows = input_cursor.fetchall()
            placeholders = ", ".join(["?"] * len(column_names))
            insert_query = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"
            output_cursor.executemany(insert_query, rows)
            output_conn.commit()

        input_conn.close()

    # Count rows in the output database
    total_rows_output = count_rows(output_conn)
    print(f"Total rows in merged database '{output_db}': {total_rows_output}")

    output_conn.close()
    print(f"Merged databases into '{output_db}' successfully.")