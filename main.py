from src.envwrapper.simulator import Simulator
from src.agent.backend import GroqModels, Provider, TogetherModels
from dotenv import load_dotenv

load_dotenv(".env")

if __name__ == '__main__':
    simulator = Simulator(
        use_db=True,
        use_gui=True,
        backend_model=TogetherModels.LLAMA31_70B,
        backend_provider=Provider.TOGETHER
    )

    # print the environments available
    print(simulator.list())

    num_simulations = 10
    # list of length num_simulations with each score (x/100)
    scores = simulator.run_multiple(
        [
       #     "single_a",
            "multi_agent_pick_item_permissions",
     #       "multi_agent_navigation",
        ],
        num_simulations
    )
    print(scores)
