import sqlite3
import pandas as pd
import re


def count_successful_simulations_and_simulations(db_path):
    """
    Counts the rows in the 'episodes' table where the score in the 'content' column is above 0,
    and counts the number of unique simulations with at least one score above 0.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        None. Prints the count of successful simulations and simulations with score > 0.
    """
    conn = sqlite3.connect(db_path)

    try:
        # Load rows where role = 'user'
        df = pd.read_sql_query("SELECT * FROM episodes WHERE role = 'user'", conn)

        if 'content' not in df.columns or 'simulation_id' not in df.columns:
            print("The required columns ('content' and 'simulation_id') do not exist in the episodes table.")
            return

        # Function to extract the score from the content
        def extract_score(content):
            match = re.search(r'The score is (\d+)', content)
            if match:
                return int(match.group(1))
            return 0  # Default to 0 if no score is found

        # Apply the score extraction to the 'content' column
        df['score'] = df['content'].apply(extract_score)

        # Filter for rows where score > 0
        successful_simulations = df[df['score'] > 0]
        row_count = len(successful_simulations)

        # Count unique simulation IDs with score > 0
        successful_simulation_count = successful_simulations['simulation_id'].nunique()

        print(f"Number of successful rows (score > 0): {row_count}")
        print(f"Number of simulations with score > 0: {successful_simulation_count}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        conn.close()


if __name__ == '__main__':
    # Usage example
    merged_db = "../uncleaned_training_data.db"  # Path to your merged database
    count_successful_simulations_and_simulations(merged_db)