import pandas as pd
import uuid


def filter_invalid_simulation_ids(df):
    """
    Filters out rows where the `simulation_id` is not a valid UUID.

    Args:
        df (pd.DataFrame): Input DataFrame containing the `simulation_id` column.

    Returns:
        pd.DataFrame: Filtered DataFrame with only valid UUIDs in the `simulation_id` column.
    """
    if 'simulation_id' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'simulation_id' column.")

    def is_valid_uuid(sim_id):
        try:
            uuid.UUID(str(sim_id))
            return True
        except (ValueError, TypeError):
            return False

    # Filter rows where simulation_id is a valid UUID
    filtered_df = df[df['simulation_id'].apply(is_valid_uuid)]

    print(f"Filtered DataFrame: {len(df) - len(filtered_df)} invalid rows removed.")
    return filtered_df