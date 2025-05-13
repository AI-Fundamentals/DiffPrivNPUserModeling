import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'models_from_csf/ex1/6400/nodp_dlv128/training_metrics.csv'
df_train = pd.read_csv(file_path)


# Create the plot
fig, ax = plt.subplots(1, 2, figsize=(8,4), sharex=True)

ax[0].plot(df_train['loss'], label='Loss', color='tab:blue')
ax[0].set_title('Training Loss')
ax[0].set_ylabel('Loss')
ax[0].set_yscale('log') 
ax[0].set_xlim(0,70)
ax[0].set_ylim(0.01)
ax[0].grid()
ax[0].set_xlabel('N of training epochs')

ax[1].plot(df_train['train_acc_greedy'], label='Train Accuracy (Greedy)', color='tab:red')
ax[1].set_title('Training Accuracy (greedy)')
ax[1].set_xlabel('N of training epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlim(0,70)
ax[1].set_ylim(0, 1)
ax[1].grid()

plt.suptitle('Experiment 1, 6400 users, dim_lv=128, c=1')

# Adjust layout and display the plots
plt.tight_layout()
plt.savefig('figures/ex1/ex1_acc_loss_vs_epochs_dimlv128.png',dpi=200)
plt.show()
