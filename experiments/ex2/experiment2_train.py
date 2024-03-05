# Load packages
import tensorflow as tf
import neuralprocesses.tensorflow as nps
import argparse
import os
import tensorflow.keras.backend as K
import numpy as np
import json
import matplotlib.pyplot as plt

from dppum.data import hdf_to_dataset_pad_tf
from dppum.loss import np_elbo_tf_cat
from dppum.util import print_dictionary
from dppum.train import train_model_dp_tf

print("Finished importing packages.")


# %%
# Parse any command line arguments

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

# Adding arguments to the parser
parser.add_argument("--num_users", 
                    help="Number of users to load from the training data hdf.", 
                    type=int, 
                    default=128)

parser.add_argument("--batch_size", 
                    help="Number of users to put into each batch.", 
                    type=int, 
                    default=32)

parser.add_argument("--train_hdf", 
                    help="The file to load the training from.", 
                    type=str,
                    default="data/ex2/experiment2_training_data.hdf")

parser.add_argument("--test_hdf", 
                    help="The file to load the training from.", 
                    type=str,
                    default="data/ex2/experiment2_test_data.hdf")

parser.add_argument("--models_dir", 
                    help="The folder to save the trained models.", 
                    type=str,
                    default="models/ex2/")

parser.add_argument("--figs_dir", 
                    help="The folder for output figures.", 
                    type=str,
                    default="figures/ex2/")

parser.add_argument("--num_samples", 
                    help="Number of samples to take for model evaluation.", 
                    type=int,
                    default=5)

parser.add_argument("--num_epochs", 
                    help="Number of training epochs.", 
                    type=int,
                    default=5)

parser.add_argument("--epsilon",
                    "--eps",
                    help="Epsilon parameter in differential privacy.", 
                    type=float,
                    default=1.0)

parser.add_argument("--clipping_bound",
                    "--cbound", 
                    help="L2 clipping bound parameter in differential privacy.", 
                    type=float,
                    default=2.0)

parser.add_argument("--learning_rate",
                    "--lr",
                    help="Learning rate for model training.", 
                    type=float,
                    default=5e-4)

# Flag for a warmup epoch (i.e. testing the untrained model)
parser.add_argument("--warmup_epoch", 
                    help="Use a warmup epoch 0 (True/False)", 
                    type=bool,
                    default=False)

# Parsing the arguments to a dictionary
args = vars(parser.parse_args())

# Save the command line args to a json in model save folder
if args['models_dir']:
    # Check if the directory exists
    if not os.path.exists(args['models_dir']):
        # If not, create the directory
        os.makedirs(args['models_dir'])
    
    # Write command line arguments to JSON file
    with open(os.path.join(args['models_dir'], "train_command_line_args.json"), 'w') as json_file:
        json.dump(args, json_file,indent=4)

print("Finished parsing command line arguments.")
# %%

# Load and prepare the data
dataset, dataset_metadata = hdf_to_tf_dataset(args['train_hdf'],dtype=tf.float32)
print(f"\nOriginal metadata for file '{args['train_hdf']}':")
print_dictionary(dataset_metadata)


dataset = dataset.shuffle(dataset_metadata['n_minibatches']).take(args['num_batches'])
dataset_metadata['n_minibatches'] = args['num_batches']
num_users = len(list(dataset)) * dataset_metadata['batch_size']
dataset_metadata['n_users'] = num_users
print(f"Dataset prepared for {num_users} users, in {args['num_batches']} (mini)batches of size {dataset_metadata['batch_size']}.")
print(f"\nUpdated metadata for file '{args['train_hdf']}':")
print_dictionary(dataset_metadata)

# Prefetch the data to make training more efficient
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print("Finished loading training dataset.")

# %%
# Clear any previous models from memory
K.clear_session()

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

print("Finished constructing the model.")

# %% 
# Train model using train_model_dp_tf function
history = train_model_dp_tf(
    model_ex2,
    dataset,
    dataset_metadata,
    loss_fn=np_elbo_tf_cat,
    num_epochs=args['num_epochs'],
    epsilon=args['epsilon'],
    clipping_bound=args['clipping_bound'],
    optimizer_name='Adam',
    learning_rate=args['learning_rate'],
    dp_enc=True,
    dp_dec=False,
    num_samples=args['num_samples'],
    warmup_epoch=args['warmup_epoch'],
    shuffle=True,
    model_save_dir = args['models_dir']
    )

print("Finished training the model.")

# %%
# Plot training metrics

# Make output folders
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

if args['warmup_epoch']:
    epochs_list = np.arange(len(history['loss']))
else:
    epochs_list = np.arange(len(history['loss'])) + 1

# Plotting the data
ax[0].plot(epochs_list, history['loss'], label='Loss')
ax[1].plot(epochs_list, history['cat_accuracy'], label='Mean Accuracy')
ax[1].plot(epochs_list, history['cat_confidence'], label='Mean Confidence')

# Adding labels and title
ax[0].set_xlabel('Epochs completed')
ax[0].set_ylabel('Loss')
ax[1].set_xlabel('Epochs completed')
ax[1].set_ylabel('Values')

# Adding legend
ax[0].legend()
ax[1].legend()

# Make loss y axis logscale
#ax[0].set_yscale('symlog')

# Display the plot when running in Spyder
plt.tight_layout()
plt.show()

# Check if the directory exists
if not os.path.exists(args['figs_dir']):
    # If not, create the directory
    os.makedirs(args['figs_dir'])

# Save the figure
fig.savefig(os.path.join(args['figs_dir'],'experiment2_training_metrics.png'))

print("Finished plotting training metrics.")

# %%
print("Finished training experiment2.py training script.")


