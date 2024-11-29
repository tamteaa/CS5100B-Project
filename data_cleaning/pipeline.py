import sqlite3
import pandas as pd
from data_cleaning.extract_successful_simulations import extract_successful_simulations_from_df
from data_cleaning.prune_non_uuid_simulations import filter_invalid_simulation_ids
from data_cleaning.format_assistant_messages import format_messages_for_fine_tuning


def print_environment_statistics_from_df(df):
    """
    Prints the number of rows for each environment in the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing an 'environment_name' column.

    Returns:
        None
    """
    if 'environment_name' not in df.columns:
        print("The DataFrame does not contain an 'environment_name' column.")
        return

    # Group by 'environment_name' and count rows
    stats = df['environment_name'].value_counts()

    # Print statistics
    print("\nEnvironment Statistics:")
    for environment, count in stats.items():
        print(f"Environment '{environment}': {count} rows")


def pipeline(input_db):
    """
    Simplified pipeline for loading a database, extracting successful simulations,
    and returning a cleaned DataFrame.

    Args:
        input_db (str): Path to the input SQLite database file.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing only successful simulations.
    """
    try:
        # Load the episodes table from the database into a DataFrame
        print(f"Reading data from '{input_db}'...")
        conn = sqlite3.connect(input_db)
        df = pd.read_sql_query("SELECT * FROM episodes", conn)
        conn.close()
        print(f"Loaded {len(df)} rows from '{input_db}'.")

        # Call the extraction function to clean the data
        print("Extracting successful simulations...")
        cleaned_df = extract_successful_simulations_from_df(df)
        print(f"Extraction complete. Cleaned data contains {len(cleaned_df)} rows.")
        cleaned_df = filter_invalid_simulation_ids(cleaned_df)

        #formatted_df = format_messages_for_fine_tuning(cleaned_df)

        # Ask the user if they want to print environment statistics
        print_stats = input(f"Do you want to print environment statistics for '{input_db}'? (yes/no): ").strip().lower()
        if print_stats in ['yes', 'y']:
            print_environment_statistics_from_df(cleaned_df)

        return cleaned_df

    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure

