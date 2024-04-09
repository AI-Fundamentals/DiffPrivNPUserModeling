
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
from dppum.loss import np_elbo_explicit
from dppum.util import print_dictionary
from dppum.torch.train import train_model_dp_torch, get_device_type_torch


# %%
# Get GPU type
device = get_device_type_torch()
B.set_global_device(device)
nps.lab.set_global_device(device)


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
                    default=4)

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
                    default=1)

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

# Cache/prefetch the data for faster training. However this will cause problems
# if the dataset is too large to fit in memory
parser.add_argument("--cache", 
                    help="Cache/Prefetch the data for faster training (True/False)", 
                    type=bool,
                    default=False)

parser.add_argument("--clip_user", 
                    help="Method to clip gradients per user. ('loop'/'vectorize'/'false')", 
                    type=str,
                    default='loop')

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
padding_values = -1.
dataset_train, metadata_train = hdf_to_dataset_pad_torch(args['train_hdf'],
                                            n_users=args['num_users'],
                                            batch_size=args['batch_size'],
                                            padding_values=padding_values
                                            )
print(f"\nMetadata for dataset from file '{args['train_hdf']}':")
print_dictionary(metadata_train)

dataset_test,metadata_test = hdf_to_dataset_pad_torch(args['test_hdf'],
                                            n_users=128,
                                            batch_size=args['batch_size'],
                                            padding_values=padding_values
                                            )
print(f"\nMetadata for dataset from file '{args['test_hdf']}':")
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
# Train model using train_model_dp_tf function
time_start = dt.datetime.now()

history = train_model_dp_torch(
    model_ex2,
    dataset_train,
    metadata_train,
    dataset_test = dataset_test,
    loss_fn=np_elbo_explicit,
    num_epochs=args['num_epochs'],
    epsilon=args['epsilon'],
    clipping_bound=args['clipping_bound'],
    optimizer_name='Adam',
    learning_rate=args['learning_rate'],
    dp_enc=True,
    dp_dec=False,
    num_samples=args['num_samples'],
    warmup_epoch=args['warmup_epoch'],
    shuffle=False,
    model_save_dir = args['models_dir'],
    padding_values=padding_values,
    clip_grads_per_user=args['clip_user']
    )

time_end = dt.datetime.now()
training_time = time_end-time_start
print("Finished training the model.")
print(f"Training time: {'{:.2f}'.format(training_time.total_seconds()/60)} minutes")