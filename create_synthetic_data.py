from src.utils.prepare_jsonl import CreateSyntheticData




if __name__ == '__main__':
    data_gen = CreateSyntheticData("agent_data.db", "episodes")

    train_file, val_file = data_gen.generate_split_data(
        training_envs=["single_agent_navigation"],
        validation_envs=["multi_agent_navigation"],
    )

