#Load local copy of neuralprocesses if present
#import sys
#sys.path.insert(0,'/Users/user/github/neuralprocesses')

import neuralprocesses.tensorflow as nps
import argparse
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import pdb
import lab as B
import numpy as np
import matplotlib.pyplot as plt


from dppum.data import hdf_to_tf_dataset
from dppum.loss import np_elbo_tf_cat
from dppum.util import print_dictionary, calc_cat_acc_onehot
from dppum.train import train_model_dp_tf

# %%

# First, parse any command line arguments

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

# Adding arguments to the parser

parser.add_argument("--num_batches", 
                    help="Number of batches to use from the training data hdf.", 
                    type=int, 
                    default=4)

parser.add_argument("--hdf", 
                    help="Directly specify the file to load the training from.", 
                    type=str,
                    default="data/ex2/experiment2_training_data.hdf")

parser.add_argument("--models_dir", 
                    help="Directly specify the folder to load the trained models from.", 
                    type=str,
                    default="models/ex2/")

parser.add_argument("--fig", 
                    help="Directly specify the folder for output figures.", 
                    type=str,
                    default="figures/ex2/")

parser.add_argument("--warmup_epoch", 
                    help="Use a warmup epoch (True/False)", 
                    type=bool,
                    default=False)

# Parsing the arguments
args = parser.parse_args()


# %%

# Load and prepare the data
dataset, dataset_metadata = hdf_to_tf_dataset(args.hdf,dtype=tf.float32)
print(f"\nOriginal metadata for file '{args.hdf}':")
print_dictionary(dataset_metadata)


dataset = dataset.shuffle(dataset_metadata['n_minibatches']).take(args.num_batches)
dataset_metadata['n_minibatches'] = args.num_batches
num_users = len(list(dataset)) * dataset_metadata['batch_size']
dataset_metadata['n_users'] = num_users
print(f"Dataset prepared for {num_users} users, in {args.num_batches} (mini)batches of size {dataset_metadata['batch_size']}.")
print(f"\nUpdated metadata for file '{args.hdf}':")
print_dictionary(dataset_metadata)

# Prefetch the data to make training more efficient
dataset = dataset.prefetch(tf.data.AUTOTUNE)



# %%
K.clear_session()

# For experiment 2:
# Data dimensions come from the data
# dim_embedding is specified in appendix as hidden dimensions
# num_enc_layers and num_dec_layers are specified in appendix as number of layers
# Nonlinearity specified in appendix
# likelihood- the anp for ex2 in Julia has a line HeterogeneousGaussianLikelihood()
# so I think that means het rather than lowrank
# dim_lv is about a latent variable for non-Gaussianity (LNP), so set to 0

model_ex2 = nps.construct_agnp(dim_x=17, dim_y=9, dim_lv=0, dim_embedding=16,
                        num_enc_layers=6,num_dec_layers=6,
                        likelihood="het",
                        nonlinearity='LeakyReLU')

print("Running with dim_embedding = 16 for speed but it should be 128")

# Number of samples taken to assess sample accuracy/confidence
num_samples=5

# Flag for a warmup epoch (i.e. testing the untrained model)
# This will be epoch 0
warmup_epoch = False

# %% 
# Train model using train_model_dp function

history = train_model_dp_tf(
    model_ex2,
    dataset,
    dataset_metadata,
    loss_fn=np_elbo_tf_cat,
    num_epochs=5,
    epsilon=1,
    clipping_bound=2,
    optimizer_name='Adam',
    learning_rate=5e-4,
    dp_enc=False,
    dp_dec=False,
    num_samples=5,
    warmup_epoch=False,
    shuffle=True,
    model_save_dir = args.models_dir
    )



# %% Plot training metrics
# Make output folders
figures_path = args.fig

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

if warmup_epoch:
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
ax[0].set_yscale('symlog')

# Display the plot when running in Spyder
plt.tight_layout()
plt.show()

# Check if the directory exists
if not os.path.exists(figures_path):
    # If not, create the directory
    os.makedirs(figures_path)

# Save the figure
fig.savefig(figures_path+'experiment2_training_metrics.png')


# %%
#testing loading a model from saved weights

iterator = iter(dataset)
xc,yc,xt,yt = next(iterator)
yc_t = B.transpose(yc,perm=[0,2,1])
yt_t = B.transpose(yt,perm=[0,2,1])



model2 = nps.construct_agnp(dim_x=17, dim_y=9, dim_lv=0, dim_embedding=16,
                        num_enc_layers=6,num_dec_layers=6,
                        likelihood="het",
                        nonlinearity='LeakyReLU')

mean, _, _, _ = nps.predict(
    model2,xc, yc_t, xt, num_samples=1
    )

acc = calc_cat_acc_onehot(yt_t,mean,cat_axis=-2)
print(f"Accuracy of untrained model: {acc}")


model2.load_weights('models/ex2/weights_epoch_5.tf')

mean, _, _, _ = nps.predict(
    model2,xc, yc_t, xt, num_samples=1
    )

acc = calc_cat_acc_onehot(yt_t,mean,cat_axis=-2)
print(f"Accuracy of loaded model: {acc}")


