import matplotlib.pyplot as plt
import pandas as pd

results = {
    "Environment Name": [
        "Single Agent Navigation",
        "Corner Multi-Agent Navigation",
        "Alphabetical Line Task",
        "Random Points Multi-Agent Navigation",
        "Single Agent Pick up item",
        "Multi Agent Pick up Item",
        "Multi-Agent Pick up item with Permissions",
    ],
    "GROQ Llama 8B": [80, 37.5, 0, 45, 20, 0, 0],  # replace with actual scores
    "GROQ Llama 70B": [90, 72.5, 62.5, 35, 90, 81.65, 74.98],
    "Fine-tuned 8B": [50, 50, 50, 50, 50, 50, 50],  # replace with actual scores
}

df_results = pd.DataFrame(results)

plt.figure(figsize=(12, 6))

x = range(len(df_results["Environment Name"]))
bar_width = 0.25

models = ["GROQ Llama 8B", "GROQ Llama 70B", "Fine-tuned 8B"]
colors = ["orange", "blue", "red"]

for i, model in enumerate(models):
    plt.bar(
        [pos + i * bar_width for pos in x],
        df_results[model],
        width=bar_width,
        label=model,
        alpha=0.8,
        color=colors[i],
    )

env_names = ["\n".join(name.split()) for name in df_results["Environment Name"]]

plt.title("Average Scores by Environment and Model", fontsize=14)
plt.xlabel("Environment Name", fontsize=12)
plt.ylabel("Average Score", fontsize=12)
plt.xticks(
    [pos + bar_width for pos in x],
    env_names,
    rotation=0,
    fontsize=10,
    ha="center",
)
plt.yticks(fontsize=10)
plt.legend(title="Models", fontsize=10)
plt.tight_layout()

plt.show()
