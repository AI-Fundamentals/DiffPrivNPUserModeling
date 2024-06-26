import neuralprocesses.torch as nps
import argparse
import os
import json
import matplotlib.pyplot as plt
import datetime as dt
import lab as B
import torch

from dppum.data import hdf_to_dataloader_pad
from dppum.loss import np_elbo_cat_torch
from dppum.util import print_dictionary
from dppum.train import train_model_dp_torch, get_device_type
from dppum.settings import default_settings_ex2_train

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
                    default="settings/settings_ex2_train.json")

# Parsing the arguments to a dictionary
settings_file_path = vars(parser.parse_args())['settings']

# Load the settings file to json
try:
    print(f"Trying to load settings from '{settings_file_path}'")
    with open(settings_file_path, 'r') as f:
        train_settings = json.load(f)
    train_settings['settings_file_path'] = settings_file_path
    print("Loaded settings successfully.")
except Exception as e:
    print("Failed to load settings due to the following error:", e)
    print("\nUsing default settings from function default_settings_ex2_train().")
    train_settings = default_settings_ex2_train()
    train_settings['settings_file_path'] = "Default"
    
# Calculate delta
if not train_settings['delta']:
    train_settings['delta'] = 1 / (train_settings['num_users'])**2

# Save the command line args to a json in model save folder
# Check if the directory exists
if not os.path.exists(train_settings['models_dir']):
    # If not, create the directory
    os.makedirs(train_settings['models_dir'])

# Save settings to models dir
with open(os.path.join(train_settings['models_dir'], "train_settings.json"), 'w') as json_file:
    json.dump(train_settings, json_file,indent=4)

print("Finished loading settings.")


# %%
# Load training and validation data
dataloader_train, metadata_train = hdf_to_dataloader_pad(train_settings['train_hdf'],
                                            n_users=train_settings['num_users'],
                                            batch_size=train_settings['batch_size'],
                                            padding_value=train_settings['padding_value']
                                            )
print(f"\nMetadata for dataloader from file '{train_settings['train_hdf']}':")
print_dictionary(metadata_train)

dataloader_val,metadata_val = hdf_to_dataloader_pad(train_settings['val_hdf'],
                                            n_users=train_settings['num_users'],
                                            batch_size=train_settings['batch_size'],
                                            padding_value=train_settings['padding_value']
                                            )
print(f"\nMetadata for dataloader from file '{train_settings['val_hdf']}':")
print_dictionary(metadata_val)

# %% Get dimensions of the data
dataiter = iter(dataloader_train)
_,_,xt,yt = next(dataiter)
dim_x = xt.shape[2]
dim_y = yt.shape[2]

# %%
# Construct the model
model_ex2 = nps.construct_agnp(
    dim_x=dim_x, # From the data dimensions
    dim_y=dim_y, # From the data dimensions
    dim_embedding=128, # Specified in appendix as hidden dimensions
    num_enc_layers=6, # Specified in appendix as number of layers
    num_dec_layers=6, # Specified in appendix as number of layers
    likelihood=train_settings['likelihood'],
    dim_lv=train_settings['dim_lv'],
    lv_likelihood=train_settings['lv_likelihood'],
    nonlinearity=train_settings['nonlinearity'],
    )
model_ex2 = model_ex2.to(device)

print("Finished constructing the model.")

# %%
# Load weights from file if required
if train_settings['init_weights']:
    model_ex2.load_state_dict(torch.load(train_settings['init_weights']))

# %%
# Setup optimizer
valid_optimizers = ['Adam']
if train_settings['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model_ex2.parameters(), lr=train_settings['learning_rate'])
else:
    raise ValueError(f"Invalid optimizer name. Expected one of: {valid_optimizers}")

# %%
# Train model using train_model_dp_torch function
time_start = dt.datetime.now()

history = train_model_dp_torch(
    model_ex2,
    dataloader_train,
    metadata_train,
    loss_fn=np_elbo_cat_torch,
    optimizer=optimizer,
    settings=train_settings,
    dataloader_val = dataloader_val
    )

time_end = dt.datetime.now()
training_time = time_end-time_start
print("Finished training the model.")
print(f"Training time: {'{:.2f}'.format(training_time.total_seconds()/60)} minutes")

# %%
# Plot training metrics

# Make output folders
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

# Plotting the data
ax[0].plot(history['epoch'], history['loss'], label='Loss')
ax[1].plot(history['epoch'], history['train_acc_greedy'], label='Train Acc (greedy)')
ax[1].plot(history['epoch'], history['train_acc_sample'], label='Train Acc (sample)')
ax[1].plot(history['epoch'], history['train_conf_greedy'], label='Train Confidence (greedy)')
if train_settings['warmup_epoch']:
    ax[1].plot(history['epoch'], history['val_acc_greedy'], label='Val Acc (greedy)')    
    ax[1].plot(history['epoch'], history['val_acc_sample'], label='Val Acc (sample)')    

# Adding labels and title
ax[0].set_xlabel('Epochs completed')
ax[0].set_ylabel('Loss')
ax[1].set_xlabel('Epochs completed')
ax[1].set_ylabel('Values')

# Adding legend
ax[0].legend()
ax[1].legend()

# Display the plot when running in Spyder
plt.tight_layout()
plt.show()

# Check if the directory exists
if not os.path.exists(train_settings['figs_dir']):
    # If not, create the directory
    os.makedirs(train_settings['figs_dir'])

# Save the figure
fig.savefig(os.path.join(train_settings['figs_dir'],'experiment2_training_metrics.png'))

print("Finished plotting training metrics.")

# %%
print("Finished training experiment2.py training script.")