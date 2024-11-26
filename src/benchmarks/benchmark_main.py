import random
from typing import Optional, Dict, Any, Callable

import pandas as pd
import os
import yaml
from numpy.ma.extras import average
from pandas.core.interchange.dataframe_protocol import DataFrame

from src.agent.backend import Provider, GroqModels
from src.envwrapper.simulator import Simulator
from src.environments.DEFAULT_CONFIGS import *


def get_default_configs():
    return DEFAULT_CONFIGS


class Benchmark:
    """
    A class to manage and execute benchmark simulations, creating the simulator internally.
    """
    BACKEND_PROVIDER = Provider.GROQ
    BACKEND_MODEL = GroqModels.LLAMA_8B

    def __init__(self, use_db: bool = False, use_gui: bool = False, output_dir: str = "./", num_simulations: int = 5,
                 backend_model: Optional[str] = GroqModels.LLAMA_8B, backend_provider: Optional[str] = Provider.GROQ):
        """
        Initializes the Benchmark class and constructs the simulator.

        :param use_db: Whether to use a database.
        :param use_gui: Whether to use a GUI.
        :param output_dir: Directory to save benchmark results.
        """
        self.configs = {}
        self.init_configs()
        self.use_db = use_db
        self.use_gui = use_gui
        self.output_dir = output_dir
        self.num_simulations = num_simulations
        os.makedirs(self.output_dir, exist_ok=True)
        self.simulator = Simulator(use_db=self.use_db, use_gui=self.use_gui, configs=self.configs,
                                   backend_model=self.BACKEND_MODEL, backend_provider=self.BACKEND_PROVIDER)
        self.termination_functions = {}
        self.BACKEND_PROVIDER = backend_provider
        self.BACKEND_MODEL = backend_model

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
            "SimNum": [0]
        })

    def run(self, config_keys: list, save_to_csv: bool = True):
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

    def run_all(self, save_to_csv: bool = True):
        """
        Runs benchmarks for all configurations.

        :param save_to_csv: Whether to save results to CSV files.
        """
        config_keys = list(self.configs.keys())
        return self.run(config_keys, save_to_csv)

    def from_config(self, config_file: str, save_to_csv: bool = True, termination_func=None):
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
        scores = self.simulator.run(config_key, num_simulations=self.num_simulations)

        stats_data = []

        envs = self.simulator.env_map
        for env_name, sims in envs.items():
            for sim_num, env in sims.items():
                for agent_id, agent in env.agents.items():
                    steps_taken = agent.variables.get("steps_taken", 0)
                    messages_sent = [
                        msg.get("content", "")
                        for msg in agent.messages
                        if msg.get("role", "") == "assistant"
                    ]

                    # Append stats for the current agent
                    stats_data.append({
                        "Agent Name": agent.name,
                        "Steps Taken": steps_taken,
                        "Score": scores[sim_num],
                        "Messages Sent": ";".join(messages_sent),
                        "SimNum": sim_num,
                    })

        # Create a DataFrame from the collected stats
        stats_df = pd.DataFrame(stats_data)

        # Calculate average steps and scores per agent
        avg_metrics = stats_df.groupby("Agent Name").agg(
            Avg_Steps=("Steps Taken", "mean"),
            Avg_Score=("Score", "mean"),
            Messages=("Messages Sent", lambda x: " || ".join(x))  # Combine all messages for each agent
        ).reset_index()

        return avg_metrics

    def build_simulation_config(self, config: Dict[str, Any], path: str, termination_func: Optional = None):
        """
        Builds a simulation config for a given configuration and collects metrics.
        :param config: The configuration to build.
        :param path: The path to the configuration file.
        :param termination_func: Function to terminate simulation after a simulation has finished.
        :return: A dictionary representing the simulation configuration.
        """
        random_vars = config.get("random_variables", {})
        num_agents = config.get("num_agents")

        # Resolve random variables
        if isinstance(num_agents, str):
            try:
                # Safely evaluate the num_agents expression
                num_agents = eval(random_vars.get("num_agents", "1"), {"random": random})
            except Exception as e:
                raise ValueError(f"Error evaluating num_agents: {e}")

        # Set default termination function
        if termination_func is None:
            if num_agents > 1:
                termination_func = align_alphabetically_task_scoring_function
            else:
                termination_func = single_agent_navigation_scoring_function

        # Return the simulation configuration
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

        default_configs = get_default_configs()

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

    def set_termination_condition(self, config_key: str, termination_condition: Callable):
        if config_key in self.configs:
            self.configs[config_key]["termination_condition"] = termination_condition
        else:
            KeyError(f"Config key {config_key} not found!")
