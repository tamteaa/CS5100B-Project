from src.envwrapper.simulator import Simulator

if __name__ == "__main__":
    simulator = Simulator(
        use_gui=True,
        use_db=True,
    )

    simulator.load_environment_config(
        [
            "single_agent_navigation.yaml",

        ]
    )

    simulator.run_all()


