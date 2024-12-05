Large Language Models (LLMs) have revolutionized decision-making, reasoning, and problem-solving, offering significant advancements in computational agents across various domains. While LLMs excel in individual tasks, their potential in
multi-agent systems(MAS), where collaboration among agents is critical—remains underexplored. This project aims to address this gap by achieving two key objectives: (1) developing multi-agent-specific benchmarks to assess the collaborative capabilities of LLMs, and (2) investigating whether collaboration can be enhanced as a trainable dimension through fine-tuning open-source LLMs. Drawing inspiration from prior studies, we propose customizing a grid environment with diverse tasks requiring agent collaboration, establishing communication protocols for inter-agent interactions, and designing a robust benchmark to evaluate performance. By fine-tuning LLMs, we seek to enhance their ability to perform collaborative tasks effectively.

Usage

Install requirements
```
pip install -r requirements.txt
```

Setting up the .env file

1. Create a .env file in the root directory. 
2. Get a Groq API key from https://console.groq.com/playground .
3. Give the key as GROQ_API_KEY1=<YOUR_API_KEY>
4. You can give multiple API keys in the above format. eg: GROQ_API_KEY2=<SECOND_API_KEY> (Optional)
5. Get a together.ai API key from https://api.together.xyz
6. Give the key as TOGETHER_API_KEY1=<YOUR_API_KEY>


```
python main.py
```

or

```python
from src.envwrapper.simulator import Simulator
from src.agent.backend import GroqModels, Provider, TogetherModels
from dotenv import load_dotenv

load_dotenv(".env")

if __name__ == '__main__':
    simulator = Simulator(
        use_db=True,
        use_gui=True,
        backend_model=GroqModels.LLAMA_8B,
        backend_provider=Provider.GROQ
    )

    # print the environments available
    print(simulator.list())

    num_simulations = 1
    # list of length num_simulations with each score (x/100)
    scores = simulator.run_multiple(
        [
            'single_agent_navigation',
#            'multi_agent_navigation',
#            'align_alphabetically_task',
#            'random_points_multi_agent_navigation',
#            'single_agent_pick_item',
#            'multi_agent_pick_item',
#            'multi_agent_pick_item_permissions'
        ],
        num_simulations
    )

```

To fine-tune the model.

1. Get an Api key from together.ai 
2. create .db files with main.py (x -> *.db)
3. run create_synthetic_data.py (*.db -> training_data.jsonl, validation_data.jsonl)
4. run run_fine_tune_job.py (training_data.jsonl, validation_data.jsonl -> fine-tuned model)
5. Follow steps on together.ai to deploy/use your fine-tuned model.

To run our fine-tuned model, it is available at https://huggingface.co/aarontamte/LLama3.1-8B-cooperative-agent-finetune