import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_loss_plots(train_csv_path, eval_csv_path):
    # Set moderate but high-quality DPI
    plt.rcParams['figure.dpi'] = 150  # for display
    plt.rcParams['savefig.dpi'] = 600  # for saved file
    plt.rcParams['lines.antialiased'] = True

    # Read the data
    train_df = pd.read_csv(train_csv_path)
    eval_df = pd.read_csv(eval_csv_path)

    # Get the loss columns
    train_loss_col = [col for col in train_df.columns if 'train/loss' in col and not col.endswith(('MIN', 'MAX'))][0]
    eval_loss_col = "togethercomputer/Meta-Llama-3.1-8B-Instruct-Reference__TOG__FT-ft-dfe1160d-881e-4202-93f9-3cb788e15c1b - eval/loss"

    # Calculate rolling average for training only (window size of 5)
    train_rolling_avg = train_df[train_loss_col].rolling(window=5, center=True).mean()

    # Calculate appropriate y-axis limits
    min_loss = min(train_df[train_loss_col].min(), eval_df[eval_loss_col].min()) * 0.95
    max_loss = max(train_df[train_loss_col].max(), eval_df[eval_loss_col].max()) * 1.05

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot training data
    ax.plot(train_df['train/global_step'], train_df[train_loss_col],
            alpha=0.3, color='#1f77b4', label='Training Loss', linewidth=1)
    ax.plot(train_df['train/global_step'], train_rolling_avg,
            color='#1f77b4', linewidth=2, label='Training Moving Average (5)')

    # Plot evaluation data
    ax.plot(eval_df['train/global_step'], eval_df[eval_loss_col],
            color='#ff7f0e', linewidth=2, label='Evaluation Loss')

    # Customize the plot
    ax.set_title('Training and Evaluation Loss', fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Training Steps', fontsize=12, weight='bold')
    ax.set_ylabel('Loss', fontsize=12, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    legend = ax.legend(fontsize=10)
    plt.setp(legend.get_texts(), weight='bold')

    # Make tick labels bold
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_weight('bold')

    # Set y-axis limits based on actual data
    ax.set_ylim(min_loss, max_loss)

    # Tight layout
    plt.tight_layout()

    # Print stats for verification
    print(f"Loss range: {min_loss:.4f} to {max_loss:.4f}")

    # Save the plot in high resolution
    plt.savefig('training_eval_loss.png', bbox_inches='tight', format='png')
    plt.show()


if __name__ == '__main__':
    train_csv_path = "train_logs.csv"
    eval_csv_path = "eval_logs.csv"
    create_loss_plots(train_csv_path, eval_csv_path)