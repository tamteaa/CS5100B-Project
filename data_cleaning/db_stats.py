import sqlite3
import pandas as pd
import re


def calculate_statistics(db_path):
    """
    Calculate descriptive statistics for the 'episodes' table in the SQLite database.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        None. Prints the statistics to the console.
    """
    conn = sqlite3.connect(db_path)

    try:
        # Load the episodes table into a DataFrame for easier analysis
        df = pd.read_sql_query("SELECT * FROM episodes", conn)

        print("General Information:")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")

        print("\nSummary of numeric columns:")
        print(df.describe())

        print("\nUnique values per column:")
        for column in df.columns:
            unique_values = df[column].nunique()
            print(f"{column}: {unique_values} unique values")

        print("\nRole distribution:")
        print(df['role'].value_counts())

        print("\nAction distribution (if any):")
        if 'action' in df.columns:
            print(df['action'].value_counts())

        print("\nEpisode statistics:")
        if 'episode_number' in df.columns:
            print(f"Total episodes: {df['episode_number'].nunique()}")
            print(f"Episode numbers range: {df['episode_number'].min()} to {df['episode_number'].max()}")

        print("\nTimestamp range:")
        if 'timestamp' in df.columns:
            print(f"First timestamp: {df['timestamp'].min()}")
            print(f"Last timestamp: {df['timestamp'].max()}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        conn.close()


def count_rows_by_environment(db_path):
    """
    Counts the number of rows for each environment_name in the 'episodes' table.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        None. Prints the counts to the console.
    """
    conn = sqlite3.connect(db_path)

    try:
        # Load the episodes table into a DataFrame
        df = pd.read_sql_query("SELECT * FROM episodes", conn)

        # Check if 'environment_name' column exists
        if 'environment_name' in df.columns:
            env_counts = df['environment_name'].value_counts()
            print("Row count for each environment_name:")
            print(env_counts)
        else:
            print("The column 'environment_name' does not exist in the episodes table.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()


def calculate_successful_data_percentage(db_path):
    """
    Calculates the percentage of rows in the 'episodes' table where the score in the 'content' column is above 0.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        None. Prints the total rows, successful rows, and their percentage.
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

        # Total rows and successful rows
        total_rows = len(df)
        successful_rows = len(df[df['score'] > 0])

        # Percentage of successful rows
        successful_percentage = (successful_rows / total_rows) * 100 if total_rows > 0 else 0

        print(f"Total rows: {total_rows}")
        print(f"Successful rows (score > 0): {successful_rows}")
        print(f"Percentage of successful rows: {successful_percentage:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        conn.close()


if __name__ == '__main__':
    # Usage example
    merged_db = "../uncleaned_training_data.db"
    calculate_successful_data_percentage(merged_db)

