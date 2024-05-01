# %% Load packages

import neuralprocesses.torch as nps
import lab as B
import argparse
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from dppum.data import hdf_to_dataloader_pad
from dppum.util import print_dictionary
from dppum.train import get_device_type
from dppum.settings import default_settings_ex2_test

print("Finished importing packages.")

# %%
# Parse the settings file from the command line argument

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

parser.add_argument("-settings", 
                    help="Path to settings json file.", 
                    type=str, 
                    default="settings/settings_ex2_test.json")

# Parsing the arguments to a dictionary
settings_file_path = vars(parser.parse_args())['settings']

# Load the settings file to json
try:
    print(f"Trying to load settings from '{settings_file_path}'")
    with open(settings_file_path, 'r') as f:
        settings = json.load(f)
    settings['settings_file_path'] = settings_file_path
    print("Loaded settings successfully.")
except Exception as e:
    print("Failed to load settings due to the following error:", e)
    print("\nUsing default settings from function default_settings_ex2_test().")
    settings = default_settings_ex2_test()
    settings['settings_file_path'] = "Default"

# Save the command line args to a json in model save folder
# Check if the directory exists
if not os.path.exists(settings['models_dir']):
    # If not, create the directory
    os.makedirs(settings['models_dir'])

# Save settings to models dir
with open(os.path.join(settings['models_dir'], "test_settings.json"), 'w') as json_file:
    json.dump(settings, json_file,indent=4)

print("Finished loading settings.")


# %% Load training arguments/metadata
training_args_path = os.path.join(args['models_dir'],'train_command_line_args.json')
training_loop_args_path = os.path.join(args['models_dir'],'training_loop_args.json')

# The command line arguments passed to the training script
with open(training_args_path, 'r') as f:
    # Load JSON data from file
    training_args = json.load(f)

# The arguments passed to the training loop function
with open(training_loop_args_path, 'r') as f:
    # Load JSON data from file
    training_loop_args = json.load(f)
    
    
# %% A list of the training epochs
if training_args['warmup_epoch']:
    epochs = np.arange(0,training_args['num_epochs']+1)
else:
    epochs = np.arange(1,training_args['num_epochs']+1)


# %% Load the test dataset

# Load and prepare the data
dataset, dataset_metadata = hdf_to_tf_dataset(args['test_hdf'],dtype=tf.float32)
print(f"\nOriginal metadata for file '{args['test_hdf']}':")
print_dictionary(dataset_metadata)

dataset = dataset.shuffle(dataset_metadata['n_minibatches']).take(args['num_batches'])
dataset_metadata['n_minibatches'] = args['num_batches']
num_users = len(list(dataset)) * dataset_metadata['batch_size']
dataset_metadata['n_users'] = num_users
print(f"Dataset prepared for {num_users} users, in {args['num_batches']} (mini)batches of size {dataset_metadata['batch_size']}.")
print(f"\nUpdated metadata for file '{args['test_hdf']}':")
print_dictionary(dataset_metadata)

# Prefetch the data to make training more efficient
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print("Finished loading test dataset.")


# %% Construct the test model
# These MUST be the same parameters as were used for training
model = nps.construct_agnp(dim_x=17, dim_y=9, dim_lv=0, dim_embedding=128,
                        num_enc_layers=6,num_dec_layers=6,
                        likelihood="het",
                        nonlinearity='LeakyReLU')


# %% Loop through epochs and test accuracy
# This loop is quite similar to the training loop but does not train the model

# Define an array to store the test accuracy for all batches in all epochs
# Dimension 0 is epochs, and dimension 1 is batches
# Epochs might start at 0 or 1 depending on if there was a warmup epoch
# Batches will always start at 0
#df_accuracy = pd.DataFrame(index=epochs,columns=np.arange(0, args['num_batches']))
df_accuracy = pd.DataFrame()
df_accuracy.index.name = "batch"

for epoch in epochs:
    # Load the trained model
    model_name = f"weights_epoch_{epoch}.tf"
    model.load_weights(os.path.join(args['models_dir'], model_name))
    accuracy_this_epoch= []   
    
    # Iterate over the batches of the dataset.
    for batch, (xc, yc, xt, yt) in enumerate(dataset):
        # Transpose the y data so they go into the model
        if(tf.rank(yc)==4):
            # This should only happen if the data are re-batched
            yc_t = B.transpose(yc,perm=[0,1,3,2])
            yt_t = B.transpose(yt,perm=[0,1,3,2])
        elif(tf.rank(yc)==3):
            yc_t = B.transpose(yc,perm=[0,2,1])
            yt_t = B.transpose(yt,perm=[0,2,1])
    
        mean, _, _, _ = nps.predict(
            model,xc, yc_t, xt, num_samples=1
            )
        
        accuracy_this_epoch.append(calc_cat_acc_onehot(yt_t,mean,cat_axis=-2).numpy())
    
    
    epoch_name = f"epoch_{epoch}"
    df_accuracy[epoch_name] = accuracy_this_epoch

# %% Save accuracy data
accuracy_csv_path = os.path.join(args['models_dir'],"accuracy_vs_epochs.csv")
df_accuracy.to_csv(accuracy_csv_path,index=True)


# %% Plot accuracy vs epochs

# Assuming 'df' is your DataFrame
means = df_accuracy.mean(axis=0)

# Calculate Q5 and Q95
q5 = df_accuracy.quantile(0.05, axis=0,numeric_only=True)
q95 = df_accuracy.quantile(0.95, axis=0,numeric_only=True)

plt.figure(figsize=(10, 6))
plt.plot(epochs, means.values, label='Mean')
plt.fill_between(epochs, q5.values, q95.values, color='b', alpha=.1, label='Q5-Q95')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.title('Mean of all batches vs Epochs with Q5 and Q95')
plt.legend()
plt.grid(True)
plt.show()
