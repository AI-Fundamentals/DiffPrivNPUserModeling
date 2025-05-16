import neuralprocesses.torch as nps
import torch
import lab as B
import argparse
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from src.data import hdf_to_dataloader_pad
from src.util import print_dictionary, calc_greedy_acc_onehot, calc_true_confidence
from src.train import get_device_type
from src.settings import default_settings_ex2_val

print("Finished importing packages.")

# %%
# Get GPU type
device = get_device_type()
B.set_global_device(device)
nps.lab.set_global_device(device)
if device == "mps":
    # Set pytorch to fallback to cpu for features where mps not available
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
print(f"Device set to '{device}'")

# %%
# Parse the settings file from the command line argument

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

parser.add_argument("-settings",
                    help="Path to settings json file.",
                    type=str,
                    default="settings/settings_ex2_val.json")

# Parsing the arguments to a dictionary
val_settings_file_path = vars(parser.parse_args())['settings']

# Load the settings file to json
try:
    print(f"Trying to load settings from '{val_settings_file_path}'")
    with open(val_settings_file_path, 'r') as f:
        val_settings = json.load(f)
    val_settings['val_settings_file_path'] = val_settings_file_path
    print("Loaded settings successfully.")
except Exception as e:
    print("Failed to load settings due to the following error:", e)
    print("\nUsing default settings from function default_settings_ex2_val().")
    val_settings = default_settings_ex2_val()
    val_settings['settings_file_path'] = "Default"

# Save the command line args to a json in model save folder
# Check if the directory exists
if not os.path.exists(val_settings['models_dir']):
    # If not, create the directory
    os.makedirs(val_settings['models_dir'])

# Save settings to models dir
with open(os.path.join(val_settings['models_dir'], "val_settings.json"), 'w') as json_file:
    json.dump(val_settings, json_file, indent=4)

print("Finished loading test settings.")

# %% Load training settings
train_settings_path = os.path.join(
    val_settings['models_dir'], 'train_settings.json')

# The command line arguments passed to the training script
with open(train_settings_path, 'r') as f:
    # Load JSON data from file
    train_settings = json.load(f)

print("Finished loading training settings.")

# %% A list of the training epochs
if train_settings['warmup_epoch']:
    epochs = np.arange(0, train_settings['num_epochs']+1)
else:
    epochs = np.arange(1, train_settings['num_epochs']+1)


# %% Load the test dataset

# Load and prepare the data
dataloader_val, metadata_val = hdf_to_dataloader_pad(val_settings['val_hdf'],
                                                     n_users=val_settings['num_users'],
                                                     batch_size=val_settings['batch_size'],
                                                     padding_value=val_settings['padding_value']
                                                     )
print(f"\nMetadata for dataloader from file '{val_settings['val_hdf']}':")
print_dictionary(metadata_val)

print("Finished loading test dataset.")

# %% Get dimensions of the data
dataiter = iter(dataloader_val)
_, _, xt, yt = next(dataiter)
dim_x = xt.shape[2]
dim_y = yt.shape[2]

# %% Construct the test model
# These MUST be the same parameters as were used for training
model = nps.construct_agnp(
    dim_x=dim_x,  # From the data dimensions
    dim_y=dim_y,  # From the data dimensions
    dim_embedding=128,  # Specified in appendix as hidden dimensions
    num_enc_layers=6,  # Specified in appendix as number of layers
    num_dec_layers=6,  # Specified in appendix as number of layers
    likelihood=train_settings['likelihood'],
    dim_lv=train_settings['dim_lv'],
    lv_likelihood=train_settings['lv_likelihood'],
    nonlinearity=train_settings['nonlinearity'],
)
model = model.to(device)

# %% Loop through epochs and test accuracy
# This loop is quite similar to the training loop but does not train the model

# Define a dataframe to store the results
columns = ["acc_greedy", "acc_sample_mean", "acc_sample_std", "acc_sample_Q5",
           "acc_sample_Q25", "acc_sample_Q50", "acc_sample_Q75", "acc_sample_Q95"]
df_results = pd.DataFrame(columns=columns, index=epochs, dtype='float')
df_results.index.name = 'epoch'

model.eval()
for epoch in epochs:
    # Load the trained model
    model_name = f"weights_epoch_{epoch}.pt"
    try:
        model.load_state_dict(torch.load(os.path.join(
            val_settings['models_dir'], model_name)))
    except:
        model.load_state_dict(torch.load(os.path.join(
            val_settings['models_dir'], model_name), map_location=torch.device('cpu')))
        model = model.to(device)

    acc_greedy_this_epoch = []
    acc_sample_this_epoch = []
    conf_greedy_this_epoch = []

    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(dataloader_val):
        # Move tensors to training device (GPU)
        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        yt = yt.to(device)

        if val_settings['padding_value']:
            # A mask for where the padding is. This is the same shape as the batch
            padding_mask = (yt == val_settings['padding_value'])
            # This is collapsed along the categorical dimension
            padding_mask = B.any(padding_mask, axis=-2)
        else:
            padding_mask = None

        # Forward pass
        with torch.no_grad():
            yt_pred, _, _, _ = nps.predict(
                model, xc, yc, xt, num_samples=train_settings['num_samples'], dtype_lik=torch.float32
            )

        # Accuracy of the non-padding values
        greedy_accuracy = calc_greedy_acc_onehot(
            yt, yt_pred, cat_axis=-2, padding_value=train_settings['padding_value'])
        # Sample accuracy is the model's confidence in the true category
        sample_accuracy = calc_true_confidence(
            yt, yt_pred, -2, padding_value=train_settings['padding_value'])

        # Append values the epoch lists
        for value in greedy_accuracy.cpu().numpy().ravel():
            acc_greedy_this_epoch.append(float(value))
        for value in sample_accuracy.cpu().numpy().ravel():
            acc_sample_this_epoch.append(float(value))

    ##### End of epoch calculations #####
    df_results.loc[epoch, 'acc_greedy'] = np.mean(acc_greedy_this_epoch)
    df_results.loc[epoch, 'acc_sample_mean'] = np.mean(acc_sample_this_epoch)
    df_results.loc[epoch, 'acc_sample_std'] = np.std(acc_sample_this_epoch)
    df_results.loc[epoch, 'acc_sample_Q5'] = np.quantile(
        acc_sample_this_epoch, 0.05)
    df_results.loc[epoch, 'acc_sample_Q25'] = np.quantile(
        acc_sample_this_epoch, 0.25)
    df_results.loc[epoch, 'acc_sample_Q50'] = np.quantile(
        acc_sample_this_epoch, 0.5)
    df_results.loc[epoch, 'acc_sample_Q75'] = np.quantile(
        acc_sample_this_epoch, 0.75)
    df_results.loc[epoch, 'acc_sample_Q95'] = np.quantile(
        acc_sample_this_epoch, 0.95)


# %% Save accuracy data
accuracy_csv_path = os.path.join(
    val_settings['models_dir'], "val_acc_vs_epochs.csv")
df_results.round(3).to_csv(accuracy_csv_path, index=True)


# %% Plot accuracy vs epochs

# Create a figure and assign it to the variable 'fig'
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_results['acc_greedy'], label='acc_greedy', linestyle='-')
ax.plot(df_results['acc_sample_Q50'], label='acc_sample_Q50', linestyle='-')
ax.fill_between(df_results.index, df_results['acc_sample_Q5'],
                df_results['acc_sample_Q95'], color='skyblue', alpha=0.4)
ax.set_xlabel('N training epochs')
ax.set_ylabel('Accuracy')
ax.legend(loc='best')
ax.grid(True)
plt.show()

# Check if the directory exists
if not os.path.exists(val_settings['figs_dir']):
    # If not, create the directory
    os.makedirs(val_settings['figs_dir'])

# Save the figure
fig.savefig(os.path.join(val_settings['figs_dir'], 'val_metrics.png'))


print("Finished plotting validation metrics.")
