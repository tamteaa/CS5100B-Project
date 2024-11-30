import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_training_plots(csv_path):
    # Read the data
    df = pd.read_csv(csv_path)

    # Get the loss column
    loss_col = [col for col in df.columns if 'train/loss' in col and not col.endswith(('MIN', 'MAX'))][0]

    # Calculate rolling average (window size of 5)
    rolling_avg = df[loss_col].rolling(window=5, center=True).mean()

    # Calculate appropriate y-axis limits
    min_loss = df[loss_col].min() * 0.95  # Add 5% padding
    max_loss = df[loss_col].max() * 1.05

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw data in light color
    ax.plot(df['train/global_step'], df[loss_col],
            alpha=0.3, color='#1f77b4', label='Loss')

    # Plot rolling average in darker color
    ax.plot(df['train/global_step'], rolling_avg,
            color='#1f77b4', linewidth=2, label='Moving Average (5)')

    # Customize the plot
    ax.set_title('Training Loss', fontsize=14, pad=20)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)

    # Set y-axis limits based on actual data
    ax.set_ylim(min_loss, max_loss)

    # Tight layout
    plt.tight_layout()

    # Print stats for verification
    print(f"Loss range: {min_loss:.4f} to {max_loss:.4f}")

    # Save the plot
    plt.savefig('training_loss_averaged.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    csv_path = "wandb_export_2024-11-29T14_05_20.311-06_00.csv"
    create_training_plots(csv_path)