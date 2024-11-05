from src.storage.table import Table

if __name__ == '__main__':
    # Example usage
    table = Table(name="Users", columns={"id": "INTEGER", "name": "TEXT", "age": "INTEGER"})

    # Generating SQL for table creation
    print(table.create_table_sql())

    # Inserting rows
    table.insert_row({"id": 1, "name": "Alice", "age": 30})
    table.insert_row({"id": 2, "name": "Bob", "age": 25})

    # Selecting rows
    rows = table.select_all()
    print("All Rows:", rows)

    # Displaying the table
    table.display()
