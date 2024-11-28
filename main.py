from src.envwrapper.simulator import Simulator
from src.agent.backend import GroqModels, Provider, TogetherModels
from dotenv import load_dotenv

load_dotenv(".env")

if __name__ == '__main__':
    simulator = Simulator(
        use_db=True,
        use_gui=True,
        backend_model=TogetherModels.LLAMA3_70B,
        backend_provider=Provider.TOGETHER
    )

    # print the environments available
    print(simulator.list())

    num_simulations = 2
    # list of length num_simulations with each score (x/100)
    scores = simulator.run_multiple(
        [
#            'single_agent_navigation',
#            'multi_agent_navigation',
#            'align_alphabetically_task',
            'random_points_multi_agent_navigation',
#            'single_agent_pick_item',
#            'multi_agent_pick_item',
#            'multi_agent_pick_item_permissions'
        ],
        num_simulations
    )
    print(scores)
