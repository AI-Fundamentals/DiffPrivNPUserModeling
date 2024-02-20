# Load packages

import neuralprocesses.tensorflow as nps
import argparse
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import json
import matplotlib.pyplot as plt


from dppum.data import hdf_to_tf_dataset
from dppum.loss import np_elbo_tf_cat
from dppum.util import print_dictionary
from dppum.train import train_model_dp_tf

print("Finished importing packages.")

# %%
# Parse any command line arguments

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

# Adding arguments to the parser
parser.add_argument("--test_hdf", 
                    help="The file to load the test from.", 
                    type=str,
                    default="data/ex2/experiment2_test_data.hdf")

parser.add_argument("--models_dir", 
                    help="The folder to load the trained models.", 
                    type=str,
                    default="models/ex2/")

parser.add_argument("--figs_dir", 
                    help="The folder for output figures.", 
                    type=str,
                    default="figures/ex2/")


# Parsing the arguments to a dictionary
args = vars(parser.parse_args())

# Save the command line args to a json in model save folder
if args['models_dir']:
    # Check if the directory exists
    if not os.path.exists(args['models_dir']):
        # If not, create the directory
        os.makedirs(args['models_dir'])
    
    # Write command line arguments to JSON file
    with open(os.path.join(args['models_dir'], "test_command_line_args.json"), 'w') as json_file:
        json.dump(args, json_file,indent=4)

print("Finished parsing command line arguments.")


# %%
# Load training metadata
training_args_path = os.path.join(args['models_dir'],'train_command_line_args.json')
training_loop_args_path = os.path.join(args['models_dir'],'training_loop_args.json')


with open(training_args_path, 'r') as f:
    # Load JSON data from file
    training_args = json.load(f)

with open(training_loop_args_path, 'r') as f:
    # Load JSON data from file
    training_loop_args = json.load(f)


