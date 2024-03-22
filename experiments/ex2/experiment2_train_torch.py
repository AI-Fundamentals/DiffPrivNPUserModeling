#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:10:38 2024

@author: mbcx5jt5
"""
import neuralprocesses.torch as nps
import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as dt

import pdb


from dppum.data import hdf_to_dataset_pad_torch
from dppum.loss import np_elbo_tf_cat
from dppum.util import print_dictionary
from dppum.train import train_model_dp_tf

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
    