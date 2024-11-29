from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    environments = [
        "Single Agent Navigation",
        "Corner Multi-Agent Navigation",
        "Alphabetical Line Task",
        "Random Points Multi-Agent Navigation",
        "Single Agent Pick up item",
        "Multi Agent Pick up Item",
        "Multi-Agent Pick up item with Permissions",
    ]
    models = ["Llama 3.1 8B", "Llama 3.1 70B", "Llama 3.1 8B Fine-tune"]

    raw_scores = [
        #8B model
        [100, 0, 100, 100, 100, 100, 0, 100, 100, 100],
        [0, 50, 50, 75, 25, 25, 25, 50, 25, 50],
        [0.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 25.0],
        [100, 0, 100, 0, 0, 0, 0, 50, 100, 100],
        [0, 100, 0, 0, 0, 0, 0, 100, 0, 0],
        [0.0, 0.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 33.3],
        # 70B model
        [100, 100, 100, 100, 100, 0, 100, 100, 100, 100],
        [50, 100, 25, 100, 75, 75, 75, 75, 75, 75],
        [0, 100, 0.0, 100.0, 0.0, 100.0, 100.0, 25.0, 100.0, 100.0],
        [0, 50, 50, 0, 100, 0, 0, 50, 0, 100],
        [100.0, 100.0, 100.0, 100.0, 100, 100, 0, 100, 100, 100],
        [100, 33.3, 66.6, 100, 75, 100, 100, 75, 66.6, 100],
        [66.6, 100, 100, 100, 33.3, 100, 50, 100, 33.3, 66.6],
        # Fine-tuned model
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [50, 100, 100, 75, 25, 75, 75, 100, 100, 100],
        [100.0, 100.0, 100.0, 50.0, 100.0, 75.0, 50.0, 50.0, 100.0, 50.0],
        [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 100.0],
        [100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 0.0, 100.0, 0.0, 0.0],
        [50.0, 0.0, 25.0, 100.0, 100.0, 0.0, 100.0, 75.0, 50.0],
        [66.66666666666666, 0.0, 0.0, 0.0, 33.33333333333333, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    data = list(product(models, environments))

    df_results = pd.DataFrame(data, columns=["Model", "Environment Name"])
    df_results["Raw Scores"] = raw_scores

    df_results["Average Score"] = df_results["Raw Scores"].apply(
        lambda scores: sum(scores) / len(scores) if len(scores) > 0 else 0
    )
    df_results["Population Std"] = df_results["Raw Scores"].apply(
        lambda scores: np.std(scores, ddof=0) if len(scores) > 0 else 0
    )

    print(df_results["Average Score"])

    plt.figure(figsize=(12, 6))

    x = np.arange(len(environments))
    bar_width = 0.25

    for i, model in enumerate(models):
        group = df_results[df_results["Model"] == model]
        plt.bar(
            x + i * bar_width,
            group["Average Score"],
            width=bar_width,
            label=model,
            alpha=0.8,
        )

    env_names = ["\n".join(name.split()) for name in environments]

    plt.title("Average Scores by Environment and Model", fontsize=14)
    plt.xlabel("Environment Name", fontsize=12)
    plt.ylabel("Average Score", fontsize=12)
    plt.xticks(
        x + bar_width,
        env_names,
        rotation=0,
        fontsize=10,
        ha="center",
    )
    plt.yticks(fontsize=10)
    plt.legend(title="Models", fontsize=10)
    plt.tight_layout()

    plt.show()
