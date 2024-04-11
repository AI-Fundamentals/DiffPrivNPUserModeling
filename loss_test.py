import sys
sys.path.insert(0,'/Users/user/github/neuralprocesses')

import neuralprocesses.torch as nps_torch
import neuralprocesses.tensorflow as nps_tf
import argparse
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as dt
import lab as B

import tensorflow as tf
import tensorflow.keras.backend as K

import torch

import pdb


from dppum.torch.data import hdf_to_dataset_pad_torch
from dppum.loss import np_elbo_explicit
from dppum.torch.loss import np_elbo_cat_torch
from dppum.util import print_dictionary
from dppum.torch.train import train_model_dp_torch, get_device_type_torch

from dppum.tf.data import hdf_to_dataset_pad_tf
from dppum.tf.loss import np_elbo_tf_cat
from dppum.util import print_dictionary
from dppum.tf.train import train_model_dp_tf

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
# Load torch dataset
padding_values = -1.
dataset_tf, metadata_tf = hdf_to_dataset_pad_tf(args['train_hdf'],
                                            n_users=args['num_users'],
                                            batch_size=args['batch_size'],
                                            padding_values=padding_values
                                            )
print(f"\nMetadata for dataset from file '{args['train_hdf']}':")
print_dictionary(metadata_tf)


# %%
# Load tensorflow dataset
padding_values = -1.
dataset_torch,metadata_torch = hdf_to_dataset_pad_torch(args['train_hdf'],
                                            n_users=args['num_users'],
                                            batch_size=args['batch_size'],
                                            padding_values=padding_values
                                            )
print(f"\nMetadata for file '{args['train_hdf']}':")
print_dictionary(metadata_torch)

# %%
# Construct the model
model_tf = nps_tf.construct_gnp(
    dim_x=17, # From the data dimensions
    dim_y=9, # From the data dimensions
    dim_embedding=128, # Specified in appendix as hidden dimensions
    num_enc_layers=6, # Specified in appendix as number of layers
    num_dec_layers=6, # Specified in appendix as number of layers
    likelihood="het", # Similar to the Julia HeterogeneousGaussianLikelihood()
    #nonlinearity='LeakyReLU' # Specified in appendix
    )

# Construct the model
model_torch = nps_torch.construct_gnp(
    dim_x=17, # From the data dimensions
    dim_y=9, # From the data dimensions
    dim_embedding=128, # Specified in appendix as hidden dimensions
    num_enc_layers=6, # Specified in appendix as number of layers
    num_dec_layers=6, # Specified in appendix as number of layers
    likelihood="het", # Similar to the Julia HeterogeneousGaussianLikelihood()
    #nonlinearity='LeakyReLU' # Specified in appendix
    )

# Make the model weights of the models the same somehow

print("Made the models")

# %%
def zero_model_weights_torch(model):
    for param in model.parameters():
        param.data.zero_()

def zero_model_weights_tf(model):
    variables_zero = [tf.zeros(shape=layer.shape) for layer in model.trainable_variables]
    model.set_weights(variables_zero)

zero_model_weights_torch(model_torch)
zero_model_weights_tf(model_tf)

def check_zero_model_weights_torch(model):
    for param in model.parameters():
        if torch.sum(param.data != 0):
            return False
    return True

zero_model_weights_torch(model_torch)
if check_zero_model_weights_torch(model_torch):
    print("All weights and biases are zero.")
else:
    print("Not all weights and biases are zero.")


# %%
import numpy as np

def check_same_weights_biases(model_torch, model_tf):
    # Get weights and biases from PyTorch model
    torch_weights = [param.data.numpy() for param in model_torch.parameters()]
    
    # Get weights and biases from TensorFlow model
    tf_weights = model_tf.get_weights()
    import pdb
    pdb.set_trace()
    # Check if the number of layers is the same
    if len(torch_weights) != len(tf_weights):
        return False
    
    # Check if weights and biases are the same for each layer
    for tw, tfw in zip(torch_weights, tf_weights):
        # Transpose TensorFlow weights if necessary
        if tw.shape != tfw.shape and tw.shape == tfw.T.shape:
            tfw = tfw.T
        if not np.allclose(tw, tfw, atol=1e-6):
            import pdb
            pdb.set_trace()
            return False
    
    return True


# Use the function
if check_same_weights_biases(model_torch, model_tf):
    print("Both models have the same weights and biases.")
else:
    print("The models do not have the same weights and biases.")
    
    
tf_weights = model_tf.get_weights()
shapes_tf = [layer.shape for layer in tf_weights]
torch_weights = [param.data.numpy() for param in model_torch.parameters()]
shapes_torch= [layer.shape for layer in torch_weights]

# %% Loop through tf dataset
subsume_context=True


for step, (xc, yc, xt, yt) in enumerate(dataset_tf):
    pdb.set_trace()
    
    state = B.global_random_state(B.dtype(xc))

    scalar_loss_tf = -B.mean(np_elbo_explicit(
            state=state,
            model=model_tf,
            contexts=[(xc,yc)],
            subsume_context=subsume_context,
            xt=xt,
            yt=yt,
            normalise=False,
            dtype_lik=tf.float32,
            num_samples=25,
            padding_values=padding_values
            ))
    #scalar_loss_tf = B.mean(vector_loss)
    print("tf_loss explicit: ",scalar_loss_tf)
    scalar_loss_tf = -B.mean(np_elbo_tf_cat(
            state=state,
            model=model_tf,
            contexts=[(xc,yc)],
            subsume_context=subsume_context,
            xt=xt,
            yt=yt,
            normalise=False,
            dtype_lik=tf.float32,
            num_samples=25,
            padding_values=padding_values
            ))
    #scalar_loss_tf = B.mean(vector_loss)
    print("tf_loss fast: ",scalar_loss_tf)
    
    xc_torch = torch.from_numpy(xc.numpy())
    yc_torch = torch.from_numpy(yc.numpy())
    xt_torch = torch.from_numpy(xt.numpy())
    yt_torch = torch.from_numpy(yt.numpy())
    
    scalar_loss_torch = -B.mean(np_elbo_explicit(
            state=state,
            model=model_torch,
            contexts=[(xc_torch,yc_torch)],
            subsume_context=subsume_context,
            xt=xt_torch,
            yt=yt_torch,
            normalise=False,
            dtype_lik=torch.float32,
            num_samples=25,
            padding_values=padding_values
            ))
    #scalar_loss_torch = B.mean(vector_loss)
    print("torch_loss explicit: ",scalar_loss_torch)
    
    scalar_loss_torch = -B.mean(np_elbo_cat_torch(
            state=state,
            model=model_torch,
            contexts=[(xc_torch,yc_torch)],
            subsume_context=subsume_context,
            xt=xt_torch,
            yt=yt_torch,
            normalise=False,
            dtype_lik=torch.float32,
            num_samples=25,
            padding_values=padding_values
            ))
    #scalar_loss_torch = B.mean(vector_loss)
    print("torch_loss fast: ",scalar_loss_torch)
    
    
    
# %% Loop through tf dataset
subsume_context=True
learning_rate = 5e-4

optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=learning_rate)
optimizer_tf = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
#optimizer_tf = tf.keras.optimizers.Adam(learning_rate=learning_rate)



for step, (xc, yc, xt, yt) in enumerate(dataset_tf):
    pdb.set_trace()
    
    state = B.global_random_state(B.dtype(xc))

    with tf.GradientTape() as encoder_tape:#, tf.GradientTape() as decoder_tape:
        scalar_loss_tf = -B.mean(np_elbo_explicit(
                state=state,
                model=model_tf,
                contexts=[(xc,yc)],
                subsume_context=subsume_context,
                xt=xt,
                yt=yt,
                normalise=False,
                dtype_lik=tf.float32,
                num_samples=25,
                padding_values=padding_values
                ))
        print("tf_loss explicit: ",scalar_loss_tf)
        #encoder_gradients = encoder_tape.gradient(scalar_loss_tf, model_tf.encoder.trainable_variables)           
        #decoder_gradients = decoder_tape.gradient(scalar_loss_tf, model_tf.decoder.trainable_variables)
        gradients = encoder_tape.gradient(scalar_loss_tf, model_tf.trainable_variables)           

    #optimizer_tf.apply_gradients(zip(encoder_gradients, model_tf.encoder.trainable_variables))    
    #optimizer_tf.apply_gradients(zip(decoder_gradients, model_tf.decoder.trainable_variables))
    optimizer_tf.apply_gradients(zip(gradients, model_tf.trainable_variables))   
    
    
    # TORCH
    xc_torch = torch.from_numpy(xc.numpy())
    yc_torch = torch.from_numpy(yc.numpy())
    xt_torch = torch.from_numpy(xt.numpy())
    yt_torch = torch.from_numpy(yt.numpy())
    
    # Reset optimizer
    optimizer_torch.zero_grad()

    
    scalar_loss_torch = -B.mean(np_elbo_explicit(
            state=state,
            model=model_torch,
            contexts=[(xc_torch,yc_torch)],
            subsume_context=subsume_context,
            xt=xt_torch,
            yt=yt_torch,
            normalise=False,
            dtype_lik=torch.float32,
            num_samples=25,
            padding_values=padding_values
            ))
    print("torch_loss fast: ",scalar_loss_torch)
    # Backward pass
    scalar_loss_torch.backward()
    optimizer_torch.step()
    
    
   
    
# %%
for step, ((xc_tf, yc_tf, xt_tf, yt_tf), (xc_torch, yc_torch, xt_torch, yt_torch)) in enumerate(zip(dataset_tf, dataset_torch)):
    # Your code here
    import pdb
    pdb.set_trace()