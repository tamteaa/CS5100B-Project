import random
from typing import Optional, Dict, Any, Callable

import pandas as pd
import os
import yaml

from src.agent.backend import Provider, GroqModels
from src.envwrapper.simulator import Simulator
from benchmark_termination_functions import *


class Benchmark:
    """
    A class to manage and execute benchmark simulations, creating the simulator internally.
    """
    BACKEND_PROVIDER = Provider.GROQ
    BACKEND_MODEL = GroqModels.LLAMA_8B

    def __init__(self, use_db: bool = False, use_gui: bool = False, output_dir: str = "./", num_simulations: int = 5, ):
        """
        Initializes the Benchmark class and constructs the simulator.

        :param use_db: Whether to use a database.
        :param use_gui: Whether to use a GUI.
        :param output_dir: Directory to save benchmark results.
        """
        self.configs = self.init_configs()
        self.use_db = use_db
        self.use_gui = use_gui
        self.output_dir = output_dir
        self.num_simulations = num_simulations
        os.makedirs(self.output_dir, exist_ok=True)
        self.simulator = Simulator(use_db=self.use_db, use_gui=self.use_gui, configs=self.configs)
        self.termination_functions = {}

    def initialize_stats_dataframe(self, agent_name: str):
        """
        Initializes a stats DataFrame for a given agent.

        :param agent_name: The name of the agent.
        :return: A pandas DataFrame with columns for metrics.
        """
        return pd.DataFrame({
            "Agent Name": [agent_name],
            "Steps Taken": [0],
            "Score": [0],
            "Messages Sent": [""],
        })

    def run(self, config_keys: list, save_to_csv: bool = False):
        """
        Runs simulations for specified configurations and collects metrics.

        :param config_keys: List of configuration keys to run simulations for.
        :param save_to_csv: Whether to save results to CSV files.
        """
        all_results = []

        for config_key in config_keys:
            print(f"Running benchmark for config: {config_key}")
            env_stats = self.run_simulation_for_config(config_key)

            if save_to_csv:
                output_path = os.path.join(self.output_dir, f"{config_key}_results.csv")
                env_stats.to_csv(output_path, index=False)
                print(f"Results saved to: {output_path}")

            all_results.append(env_stats)

        return all_results

    def run_all(self, save_to_csv: bool = False):
        """
        Runs benchmarks for all configurations.

        :param save_to_csv: Whether to save results to CSV files.
        """
        config_keys = list(self.configs.keys())
        return self.run(config_keys, save_to_csv)

    def from_config(self, config_file: str, save_to_csv: bool = False, termination_func=None):
        """
        Loads a single configuration from a file and runs the benchmark.

        :param config_file: Path to the configuration file.
        :param save_to_csv: Whether to save results to a CSV file.
        :param termination_func: Function to terminate simulation after a simulation has finished.
        """
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        config_key = os.path.basename(config_file).split(".")[0]
        if config_key not in self.configs:
            self.configs[config_key] = self.build_simulation_config(config, config_file, termination_func)
            self.simulator.configs = self.configs

        return self.run([config_key], save_to_csv)

    def run_simulation_for_config(self, config_key: str):
        """
        Runs a single simulation for a given configuration and collects metrics.

        :param config_key: The key of the configuration to run.
        :return: A pandas DataFrame with collected stats.
        """
        # Run the simulation
        scores = self.simulator.run(config_key, num_simulations=self.num_simulations)

        # Collect environment and agent stats
        env = self.simulator.load_environment_config(config_key)
        stats_dataframes = []

        for agent_id, agent in env.agents.items():
            agent_name = agent.name
            stats_df = self.initialize_stats_dataframe(agent_name)

            # Fill stats from simulation
            stats_df["Steps Taken"] = agent.variables.get("steps_taken", 0)
            stats_df["Score"] = agent.variables.get("score", 0)
            stats_df["Messages Sent"] = ";".join(msg.get("content") for msg in agent.messages)

            stats_dataframes.append(stats_df)

        # Concatenate stats for all agents
        return pd.concat(stats_dataframes, ignore_index=True)

    def build_simulation_config(self, config: Dict[str, Any], path: str, termination_func: Optional):
        """
        Builds a simulation config for a given configuration and collects metrics.
        :param config: The configuration to build.
        :param path: The path to the configuration file.
        :param termination_func: Function to terminate simulation after a simulation has finished.
        """
        rand_num_agents = config.get("random_variables", {}).get("num_agents", None)
        num_agents = config.get("num_agents")
        if rand_num_agents is not None and rand_num_agents == num_agents:
            num_agents = eval(num_agents, {"random": random})
        if termination_func is None:
            if num_agents > 1:
                termination_func = align_alphabetically_task_scoring_function
            else:
                termination_func = single_agent_navigation_scoring_function
        return {
            "yaml_file": path,
            "termination_condition": termination_func,
            "backend_provider": self.BACKEND_PROVIDER,
            "backend_model": self.BACKEND_MODEL,
        }

    def init_configs(self):
        """
        Initializes the configs dictionaries.
        :return: A dictionary of configs.
        """
        file_directory = os.path.dirname(__file__)

        configs_directory = os.path.join(file_directory, "..", "..", "configs")

        if not os.path.exists(configs_directory):
            raise FileNotFoundError(f"Config directory not found: {configs_directory}")

        default_configs = self.get_default_configs()

        for filename in os.listdir(configs_directory):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                config_file = os.path.join(configs_directory, filename)

                with open(config_file, "r") as file:
                    config = yaml.safe_load(file)

                config_key = os.path.basename(config_file).split(".")[0]
                if config_key in default_configs:
                    config = default_configs[config_key]
                    config["yaml_file"] = config_file
                    self.configs[config_key] = config
                    continue

                self.configs[config_key] = self.build_simulation_config(
                    config, config_file
                )

    def get_default_configs(self):
        return {
            "single_agent_navigation": {
                "yaml_file": "single_agent_navigation",
                "termination_condition": single_agent_navigation_scoring_function,
                "backend_provider": self.BACKEND_PROVIDER,
                "backend_model": self.BACKEND_MODEL,
            },
            "multi_agent_navigation": {
                "yaml_file": "multi_agent_navigation",
                "termination_condition": multi_agent_navigation_scoring_function,
                "backend_provider": self.BACKEND_PROVIDER,
                "backend_model": self.BACKEND_MODEL,
            },
            "alphabetical_order": {
                "yaml_file": "alphabetical_order",
                "termination_condition": align_alphabetically_task_scoring_function,
                "backend_provider": self.BACKEND_PROVIDER,
                "backend_model": self.BACKEND_MODEL,
            },
            "random_points_multi_agent_navigation": {
                "yaml_file": "random_points_multi_agent_navigation",
                "termination_condition": random_points_multi_agent_navigation_scoring_function,
                "backend_provider": self.BACKEND_PROVIDER,
                "backend_model": self.BACKEND_MODEL,
            },
            "single_agent_pick_item": {
                "yaml_file": "single_agent_pick_item",
                "termination_condition": single_agent_pick_item_scoring_function,
                "backend_provider": self.BACKEND_PROVIDER,
                "backend_model": self.BACKEND_MODEL,
            },
            "multi_agent_pick_item": {
                "yaml_file": "multi_agent_pick_item",
                "termination_condition": multi_agent_pick_item_scoring_function,
                "backend_provider": self.BACKEND_PROVIDER,
                "backend_model": self.BACKEND_MODEL,
            },
        }

    def set_termination_condition(self, config_key: str, termination_condition: Callable):
        if config_key in self.configs:
            self.configs[config_key]["termination_condition"] = termination_condition
        else:
            KeyError(f"Config key {config_key} not found!")
