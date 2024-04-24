import neuralprocesses.torch as nps
import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as dt
import lab as B

import pdb

from dppum.torch.data import hdf_to_dataset_pad_torch
from dppum.torch.loss import np_elbo_cat_torch
from dppum.util import print_dictionary
from dppum.torch.train import train_model_dp_torch, get_device_type
from dppum.settings import default_settings_ex2

# %%
# Get GPU type
device = get_device_type()
B.set_global_device(device)
nps.lab.set_global_device(device)


# %%
# Parse any command line arguments

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

parser.add_argument("-settings", 
                    help="Path to settings json file.", 
                    type=str, 
                    default="settings_ex2.json")

# Parsing the arguments to a dictionary
settings_file_path = vars(parser.parse_args())['settings']
# Load the settings file to json
try:
    settings = json.load(settings_file_path)
    settings['settings_file_path'] = settings_file_path
except:
    settings = default_settings_ex2()
    settings['settings_file_path'] = "Default"


# Save the command line args to a json in model save folder
# Check if the directory exists
if not os.path.exists(settings['models_dir']):
    # If not, create the directory
    os.makedirs(settings['models_dir'])

# Save settings to models dir
with open(os.path.join(settings['models_dir'], "train_settings.json"), 'w') as json_file:
    json.dump(settings, json_file,indent=4)

print("Finished loading settings.")


# %%
settings['warmup_epoch'] = True
padding_values = -1.
dataset_train, metadata_train = hdf_to_dataset_pad_torch(settings['train_hdf'],
                                            n_users=settings['num_users'],
                                            batch_size=settings['batch_size'],
                                            padding_values=settings['padding_values']
                                            )
print(f"\nMetadata for dataset from file '{settings['train_hdf']}':")
print_dictionary(metadata_train)

dataset_test,metadata_test = hdf_to_dataset_pad_torch(settings['test_hdf'],
                                            n_users=settings['num_users'],
                                            batch_size=settings['batch_size'],
                                            padding_values=settings['padding_values']
                                            )
print(f"\nMetadata for dataset from file '{settings['test_hdf']}':")
print_dictionary(metadata_test)


# %%

# Construct the model
model_ex2 = nps.construct_agnp(
    dim_x=17, # From the data dimensions
    dim_y=9, # From the data dimensions
    dim_embedding=128, # Specified in appendix as hidden dimensions
    num_enc_layers=6, # Specified in appendix as number of layers
    num_dec_layers=6, # Specified in appendix as number of layers
    likelihood="het", # Similar to the Julia HeterogeneousGaussianLikelihood()
    nonlinearity='LeakyReLU' # Specified in appendix
    )

model_ex2 = model_ex2.to(device)

print("Finished constructing the model.")

# %%
# Train model using train_model_dp_torch function
time_start = dt.datetime.now()

history = train_model_dp_torch(
    model_ex2,
    dataset_train,
    metadata_train,
    dataset_test = dataset_test,
    loss_fn=np_elbo_cat_torch,
    num_epochs=settings['num_epochs'],
    epsilon=settings['epsilon'],
    clipping_bound=settings['clipping_bound'],
    optimizer_name=settings['optimizer'],
    learning_rate=settings['learning_rate'],
    dp_enc=settings['dp_enc'],
    dp_dec=settings['dp_dec'],
    num_samples=settings['num_samples'],
    warmup_epoch=settings['warmup_epoch'],
    shuffle=settings['shuffle'],
    model_save_dir = settings['models_dir'],
    padding_values=settings['padding_values'],
    clip_grads_per_user=settings['clip_user']
    )

time_end = dt.datetime.now()
training_time = time_end-time_start
print("Finished training the model.")
print(f"Training time: {'{:.2f}'.format(training_time.total_seconds()/60)} minutes")