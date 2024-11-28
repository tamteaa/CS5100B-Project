import pandas as pd
import json


def format_messages_for_fine_tuning(df):
    """
    Formats the 'content' column of assistant messages in the DataFrame to be better suited for fine-tuning.

    Args:
        df (pd.DataFrame): Input DataFrame with a `role` column and `content` containing JSON-formatted messages.

    Returns:
        pd.DataFrame: DataFrame with formatted assistant messages.
    """

    def reformat_content(row):
        if row['role'] == 'assistant':
            try:
                # Parse the JSON content
                message_data = json.loads(row['content'])

                # Reformat the content into a better structure
                formatted_content = (
                    f"Reflection: {message_data['reflection']}\n"
                    f"Rationale: {message_data['rationale']}\n"
                    f"Action: {message_data['action_name']} (parameters: {message_data['action_parameters']})\n"
                    f"Message: {message_data['message']}\n"
                    f"Memory: {message_data['add_memory']}"
                )
                return formatted_content
            except (KeyError, json.JSONDecodeError):
                # Return the original content if formatting fails
                return row['content']
        else:
            # Leave non-assistant rows unchanged
            return row['content']

    # Apply the reformatting to the content column
    df['content'] = df.apply(reformat_content, axis=1)
    return df