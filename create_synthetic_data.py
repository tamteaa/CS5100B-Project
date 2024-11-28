from src.utils.prepare_jsonl import CreateSyntheticData
from data_cleaning.pipeline import pipeline
from data_cleaning.merge_dbs import merge_databases


def merge_dbs():
    if __name__ == '__main__':
        db_files = [
            "/raw_data/aaron_unfiltered.db",
            "/raw_data/random_points_multi_agent_navigation.db",
            "/raw_data/Sebastian_MultiAgent_agent_data.db",
            "/raw_data/sudhanva_alphabetical_model_70_batch_1.db",
            "/raw_data/sudhanva_alphabetical_model_70_batch_2.db",
            "/raw_data/sudhanva_alphabetical_model_90_batch_1.db",
            "/raw_data/sudhanva_alphabetical_model_90_batch_2.db"
        ]

        merge_databases(
            input_dir="raw_data",
            output_db="merged.db"
        )


def process_data():
    input_db = "merged.db"
    result_df = pipeline(input_db)

    creator = CreateSyntheticData(df=result_df)
    creator.generate_split_data(
        training_envs=[
         #   "single_agent_navigation",
            "multi_agent_navigation",
          #  "single_agent_pick_item",
            "multi_agent_pick_item",
            "align_alphabetically_task",
            "alphabetical_order"
        ],
        validation_envs=[
            "multi_agent_pick_item_permissions",
            "random_points_multi_agent_navigation",
        ],
    )


if __name__ == '__main__':
    merge_dbs()

    process_data()