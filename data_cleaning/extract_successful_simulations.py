import pandas as pd
import re


def extract_successful_simulations_from_df(input_df):
    """
    Extracts all messages from simulations where the score in a user message is above 0
    and returns the cleaned DataFrame.

    Args:
        input_df (pd.DataFrame): Input DataFrame containing the episodes table.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only successful simulations.
    """
    try:
        if 'content' not in input_df.columns or 'simulation_id' not in input_df.columns:
            raise ValueError("The required columns ('content' and 'simulation_id') do not exist in the input DataFrame.")

        # Extract scores from user messages
        def extract_score(content):
            match = re.search(r'The score is (\d+)', content)
            if match:
                return int(match.group(1))
            return 0

        input_df['score'] = input_df['content'].apply(extract_score)

        # Find simulation IDs where the score > 0
        successful_sim_ids = input_df.loc[
            (input_df['role'] == 'user') & (input_df['score'] > 0),
            'simulation_id'
        ].unique()

        # Filter all rows for successful simulations
        cleaned_df = input_df[input_df['simulation_id'].isin(successful_sim_ids)]

        print(f"Extracted {len(cleaned_df)} rows from {len(successful_sim_ids)} successful simulations.")
        return cleaned_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure