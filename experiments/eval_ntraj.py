# SPDX-FileCopyrightText: 2025 University of Manchester
#
# SPDX-License-Identifier: apache-2.0

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
from src.settings import default_settings_ex2_eval_ntraj

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
                    default="settings/ex2/settings_ex2_eval_ntraj.json")

# Parsing the arguments to a dictionary
eval_settings_file_path = vars(parser.parse_args())['settings']

# Load the settings file to json
try:
    print(f"Trying to load settings from '{eval_settings_file_path}'")
    with open(eval_settings_file_path, 'r') as f:
        eval_settings = json.load(f)
    eval_settings['eval_settings_file_path'] = eval_settings_file_path
    print("Loaded settings successfully.")
except Exception as e:
    print("Failed to load settings due to the following error:", e)
    print("\nUsing default settings from function default_settings_ex2_eval().")
    eval_settings = default_settings_ex2_eval_ntraj()
    eval_settings['settings_file_path'] = "Default"

print("Finished loading test settings.")

# %% Check eval_settings is in list of valid experiments
valid_experiments = {1, 2, 3}
if eval_settings['experiment'] not in valid_experiments:
    raise ValueError(
        f"Invalid experiment value in eval settings. Must be one of {valid_experiments}.")

# %% Load training settings
train_settings_path = os.path.join(
    eval_settings['models_dir'], 'train_settings.json')

# The command line arguments passed to the training script
with open(train_settings_path, 'r') as f:
    # Load JSON data from file
    train_settings = json.load(f)

print("Finished loading training settings.")

# %%
# Save the command line args to a json in model save folder
# Check if the directory exists
if not os.path.exists(eval_settings['models_dir']):
    # If not, create the directory
    os.makedirs(eval_settings['models_dir'])

# Save settings to models dir
with open(os.path.join(eval_settings['models_dir'], "eval_ntraj_settings.json"), 'w') as json_file:
    json.dump(eval_settings, json_file, indent=4)

# %% Load the test dataset

# Load and prepare the data
dataloader_eval, metadata_eval = hdf_to_dataloader_pad(eval_settings['eval_hdf'],
                                                       n_users=eval_settings['num_users'],
                                                       batch_size=eval_settings['batch_size'],
                                                       padding_value=eval_settings['padding_value']
                                                       )
print(f"\nMetadata for dataloader from file '{eval_settings['eval_hdf']}':")
print_dictionary(metadata_eval)

if metadata_eval['n_traj'] != 10:
    raise ValueError("HDF file for n_traj evaluation must have n_traj=10.")

print("Finished loading test dataset.")

# %% Get dimensions of the data
print("Getting dimensions of the data.")
dataiter = iter(dataloader_eval)
_, _, xt, yt = next(dataiter)
dim_x = xt.shape[2]
dim_y = yt.shape[2]

print("Finished getting dimensions of the data.")

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

# %% Load the model weights
# Load weights from file if required
if eval_settings['init_weights']:
    try:
        # This works if training on the same device type as evaluation
        model.load_state_dict(torch.load(eval_settings['init_weights']))
    except:
        # This works e.g. if the model was trained on CUDA but is now running on mps on a Mac
        model.load_state_dict(torch.load(
            eval_settings['init_weights'], map_location=torch.device('cpu')))
        model = model.to(device)
else:
    raise ValueError(
        "Training settings file must contain the init_weights key with a value of the path to the file from which to load model weights.")

# %% Loop through ntraj and test accuracy
# This loop is quite similar to the training loop but does not train the model

# Define a dataframe to store the results
columns = ["acc_greedy", "acc_sample_mean", "acc_sample_Q5",
           "acc_sample_Q25", "acc_sample_Q50", "acc_sample_Q75", "acc_sample_Q95"]
n_traj = np.arange(0, 11)
df_results = pd.DataFrame(columns=columns, index=n_traj, dtype='float')
df_results.index.name = 'n_traj'

model.eval()

for ntraj in n_traj:
    print(f"Running for n_traj = {ntraj}")

    # Dimensions for cropping the data. Assumes dimension 0 is the batch dimension.
    # Dimension 3 is then the length of the context trajectory (made up of
    # multiple context trajectories).
    if ntraj == 0:
        start_dim3 = -1
    elif eval_settings['experiment'] == 1:
        start_dim3 = 10*(10-ntraj)
    elif eval_settings['experiment'] == 2:
        start_dim3 = 50 - 5*ntraj
    elif eval_settings['experiment'] == 3:
        start_dim3 = 200 - 20*ntraj

    acc_greedy_this_ntraj = []
    acc_sample_this_ntraj = []
    conf_greedy_this_ntraj = []

    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(dataloader_eval):

        # Move tensors to training device (GPU)
        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        yt = yt.to(device)

        # Crop tensors to the right ntraj
        xc = xc[:, :, :, start_dim3:]
        yc = yc[:, :, :, start_dim3:]

        # Forward pass
        with torch.no_grad():
            yt_pred, _, _, _ = nps.predict(
                model, xc, yc, xt, num_samples=train_settings['num_samples'], dtype_lik=torch.float32
            )

        # Accuracy of the non-padding values
        greedy_accuracy = calc_greedy_acc_onehot(
            yt, yt_pred, cat_axis=-2, padding_value=eval_settings['padding_value'])
        # Sample accuracy is the model's confidence in the true category
        sample_accuracy = calc_true_confidence(
            yt, yt_pred, -2, padding_value=eval_settings['padding_value'])

        # Append values the lists
        for value in greedy_accuracy.cpu().numpy().ravel():
            acc_greedy_this_ntraj.append(float(value))
        for value in sample_accuracy.cpu().numpy().ravel():
            acc_sample_this_ntraj.append(float(value))

    ##### End of iternation calculations #####
    df_results.loc[ntraj, 'acc_greedy'] = np.mean(acc_greedy_this_ntraj)
    df_results.loc[ntraj, 'acc_sample_mean'] = np.mean(acc_sample_this_ntraj)
    df_results.loc[ntraj, 'acc_sample_Q5'] = np.quantile(
        acc_sample_this_ntraj, 0.05)
    df_results.loc[ntraj, 'acc_sample_Q25'] = np.quantile(
        acc_sample_this_ntraj, 0.25)
    df_results.loc[ntraj, 'acc_sample_Q50'] = np.quantile(
        acc_sample_this_ntraj, 0.5)
    df_results.loc[ntraj, 'acc_sample_Q75'] = np.quantile(
        acc_sample_this_ntraj, 0.75)
    df_results.loc[ntraj, 'acc_sample_Q95'] = np.quantile(
        acc_sample_this_ntraj, 0.95)


# %% Save accuracy data
accuracy_csv_path = os.path.join(
    eval_settings['models_dir'], "eval_acc_vs_ntraj.csv")
df_results.round(3).to_csv(accuracy_csv_path, index=True)

# %% Plot accuracy vs ntraj

# Create a figure and assign it to the variable 'fig'
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_results['acc_greedy'], label='acc_greedy', linestyle='-')
ax.plot(df_results['acc_sample_Q50'], label='acc_sample_Q50', linestyle='-')
ax.fill_between(df_results.index, df_results['acc_sample_Q25'],
                df_results['acc_sample_Q75'], color='skyblue', alpha=0.4)
ax.set_xlabel('N of context trajectories')
ax.set_ylabel('Accuracy')
ax.legend(loc='best')
ax.grid(True)
plt.show()

# Check if the directory exists
if not os.path.exists(eval_settings['figs_dir']):
    # If not, create the directory
    os.makedirs(eval_settings['figs_dir'])

# Save the figure
fig.savefig(os.path.join(eval_settings['figs_dir'], 'eval_acc_vs_ntraj.png'))

print("Finished plotting n_traj eval metrics.")
