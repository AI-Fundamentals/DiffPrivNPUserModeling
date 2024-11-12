import pandas as pd
import matplotlib.pyplot as plt
import os

# List of file paths
file_paths = [
    'models_from_csf/ex1/6400/e1_dlv0/training_metrics.csv',
    'models_from_csf/ex1/6400/e5_dlv0/training_metrics.csv',
    'models_from_csf/ex1/6400/e10_dlv0/training_metrics.csv',
    'models_from_csf/ex1/6400/nodp_dlv0/training_metrics.csv'
]

# List of colors for the plots
colors = ['tab:red', 'tab:green', 'tab:blue', 'k']
labels = ["eps 1", "eps 5", "eps 10", "no DP"]

# Initialize the plot
fig, ax = plt.subplots()

# Loop through each file path, load the CSV if it exists, and plot the 'train_acc_sample' column
for file_path, color, label in zip(file_paths, colors, labels):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col='epoch')
        ax.plot(df.index, df['train_acc_sample'], label=label, color=color)
    else:
        print(f"Warning: {file_path} does not exist and will be skipped.")

# Add labels, title, and legend
ax.set_xlabel('N of training epochs')
ax.set_ylabel('Train Accuracy Sample, 6400 users')
ax.set_title('Train Accuracy Sample over Epochs')
ax.legend()

# Set y-axis limits
ax.set_ylim(0.1, 1)
ax.set_xlim(0, 70)

# Add dashed gridlines every 5 on the x-axis and every 0.1 on the y-axis
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.grid(which='both', linestyle='--')

# Make the plot square by setting equal aspect ratio
fig.set_figwidth(5)
fig.set_figheight(5)

# Show the plot
plt.tight_layout()
plt.show()
