import pandas as pd
import matplotlib.pyplot as plt
import os

# List of file paths
folder_list = [
    'models_from_csf/ex1/6400/e1_dlv0_c2/',
    'models_from_csf/ex1/6400/e5_dlv0_c2/',
    'models_from_csf/ex1/6400/e10_dlv0_c2/',
    'models_from_csf/ex1/6400/nodp_dlv0/'
]

# List of colors for the plots
colors = ['firebrick', 'limegreen', 'blue', 'k']
labels = ["eps 1", "eps 5", "eps 10", "non-DP"]

# %% Plot of just accuracy
# Initialize the plot
fig, ax = plt.subplots(1, 2, figsize=(8, 5))

# Loop through each file path, load the CSV if it exists, and plot the 'train_acc_sample' column
for folder_path, color, label in zip(folder_list, colors, labels):
    file_path_val = folder_path + 'val_acc_vs_epochs.csv'
    if os.path.exists(file_path_val):
        df = pd.read_csv(file_path_val, index_col='epoch')
        ax[0].plot(df.index, df['acc_greedy'], label=label, color=color)
        ax[1].plot(df.index, df['acc_sample_Q50'], label=label, color=color)
        ax[1].fill_between(df.index, df['acc_sample_Q25'],
                           df['acc_sample_Q75'], color=color, alpha=0.1)

    else:
        print(f"Warning: {file_path_val} does not exist and will be skipped.")

# Add labels, title, and legend
ax[0].set_xlabel('N of training epochs')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Greedy accuracy')
ax[0].legend(loc='lower right')
ax[1].set_xlabel('N of training epochs')
ax[1].set_title('Sample accuracy')
ax[1].legend(loc='lower right')

# Set y-axis limits
ax[0].set_ylim(0., 1)
ax[0].set_xlim(0, 70)
ax[1].set_ylim(0., 1)
ax[1].set_xlim(0, 70)

# Add gridlines
ax[0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax[0].yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax[0].grid(which='both', linestyle='--')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(10))
ax[1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax[1].grid(which='both', linestyle='--')

plt.suptitle('Experiment 1, 6400 users, dim_lv=0, c=2')

plt.tight_layout()
plt.savefig('figures/ex1/ex1_acc_vs_epochs.png', dpi=200)
plt.show()

# %% Same plot but with training loss as well
# Initialize the plot
fig, ax = plt.subplots(1, 3, figsize=(8, 4))

# Loop through each file path, load the CSV if it exists, and plot the 'train_acc_sample' column
for folder_path, color, label in zip(folder_list, colors, labels):
    file_path_val = folder_path + 'val_acc_vs_epochs.csv'
    file_path_train = folder_path + 'training_metrics.csv'
    if os.path.exists(file_path_val):
        df = pd.read_csv(file_path_val, index_col='epoch')
        ax[1].plot(df.index, df['acc_greedy'], label=label, color=color)
        ax[2].plot(df.index, df['acc_sample_Q50'], label=label, color=color)
        ax[2].fill_between(df.index, df['acc_sample_Q25'],
                           df['acc_sample_Q75'], color=color, alpha=0.1)
    else:
        print(f"Warning: {file_path_val} does not exist and will be skipped.")

    if os.path.exists(file_path_train):
        df = pd.read_csv(file_path_train, index_col='epoch')
        ax[0].plot(df.index, df['loss'], label=label, color=color)
    else:
        print(f"Warning: {file_path_val} does not exist and will be skipped.")

# Add labels, title, and legend
ax[0].set_xlabel('N of training epochs')
ax[0].set_title('Training loss')
ax[0].legend(loc='upper right')
ax[1].set_xlabel('N of training epochs')
ax[1].set_title('Greedy accuracy')
ax[1].legend(loc='lower right')
ax[2].set_xlabel('N of training epochs')
ax[2].set_title('Sample accuracy')
ax[2].legend(loc='lower right')

# Set y-axis limits
ax[0].set_ylim(0., 0.85)
ax[0].set_xlim(0, 70)
ax[1].set_ylim(0., 1)
ax[1].set_xlim(0, 70)
ax[2].set_ylim(0., 1)
ax[2].set_xlim(0, 70)

# Add gridlines
ax[0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax[0].yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax[0].grid(which='both', linestyle='--')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(10))
ax[1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax[1].grid(which='both', linestyle='--')
ax[2].xaxis.set_major_locator(plt.MultipleLocator(10))
ax[2].yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax[2].grid(which='both', linestyle='--')

plt.suptitle('Experiment 1, 6400 users, dim_lv=0, c=2')

plt.tight_layout()
plt.savefig('figures/ex1/ex1_acc_loss_vs_epochs.png', dpi=200)
plt.show()
