#Load local copy of neuralprocesses
import sys
sys.path.insert(0,'/Users/user/github/neuralprocesses')
sys.path.insert(0,'/Users/user/github/stheno')

import neuralprocesses.tensorflow as nps
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
from dppum.data import hdf_to_tf_dataset, hdf_get_metadata
from dppum.loss import np_elbo_tf_cat, np_elbo_explicit
from dppum.util import calc_cat_confidence, flatten_first_two_dims
import pdb
import lab as B
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %%

# First, parse any command line arguments

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

# Adding arguments to the parser

parser.add_argument("--num_batches", 
                    help="Number of batches to use from the training data hdf.", 
                    type=int, 
                    default=192)

parser.add_argument("--hdf", 
                    help="Directly specify the file to load the training from.", 
                    type=str,
                    default="data/ex2/experiment2_training_data.hdf")

parser.add_argument("--models", 
                    help="Directly specify the folder to load the trained models from.", 
                    type=str,
                    default="data/ex2/experiment2_training_data.hdf")

parser.add_argument("--fig", 
                    help="Directly specify the folder for output figures.", 
                    type=str,
                    default="figures/ex2/")

# Parsing the arguments
args = parser.parse_args()


# %%

# Load and prepare the data
data = hdf_to_tf_dataset(args.hdf,dtype=tf.float32)
metadata = hdf_get_metadata(args.hdf)
print(metadata)

num_batches = 4  # replace with your desired number of batches
data = data.shuffle(data.cardinality()).take(num_batches)
num_users = len(list(data)) * metadata['batch_size']
print(f"Data loaded for {num_users} users.")

# Prefetch the data to make training more efficient
data = data.prefetch(tf.data.AUTOTUNE)



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

model = nps.construct_agnp(dim_x=17, dim_y=9, dim_lv=0, dim_embedding=16,
                        num_enc_layers=6,num_dec_layers=6,
                        likelihood="het",
                        nonlinearity='LeakyReLU') #This improves training if true

print("Running with dim_embedding = 16 for speed but it should be 128")

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)




# Number of samples taken to assess sample accuracy/confidence
num_samples=5

# Metrics to keep track of training within epochs
per_epoch_loss = tf.keras.metrics.Mean(name='elbo_loss')
per_epoch_mean_acc = tf.keras.metrics.Mean(name='mean_acc')
per_epoch_mean_conf = tf.keras.metrics.Mean(name='mean_confidence')
per_epoch_sample_acc = tf.keras.metrics.Mean(name='sample_accuracy')
per_epoch_sample_conf = tf.keras.metrics.Mean(name='sample_conf')

# Keep track of training metrics per epoch
epochs_list = []
epochs_loss = []
epochs_mean_acc = []
epochs_mean_conf = []
epochs_sample_acc = []
epochs_sample_conf = []

# Metrics to keep track of training within batches
per_batch_loss = tf.keras.metrics.Mean(name='elbo_loss')
per_batch_mean_acc = tf.keras.metrics.Mean(name='mean_acc')
per_batch_mean_conf = tf.keras.metrics.Mean(name='mean_confidence')
per_batch_sample_acc = tf.keras.metrics.Mean(name='sample_accuracy')
per_batch_sample_conf = tf.keras.metrics.Mean(name='sample_conf')

# Keep track of training metrics per task
batches_list = []
batches_loss = []
batches_mean_acc = []
batches_mean_conf = []
batches_sample_acc = []
batches_sample_conf = []

# Flag for a warmup epoch (i.e. testing the untrained model)
# This will be epoch 0
warmup_epoch = False
if warmup_epoch:
    first_epoch = 0
else:
    first_epoch = 1

# Fix the noise for the early epochs to force the model to fit.
fixed_sigma_epochs = -1
fixed_sigma = 1e-2

# %%
# Training loop
# The first training epoch is epoch 1
num_epochs = 5









for epoch in range(first_epoch,num_epochs+1):
    print(f"""######## Start of epoch {epoch} ########""")
    
    # Shuffle the dataset
    data = data.shuffle(buffer_size=num_users)
    
    # Reset epoch metrics
    per_epoch_loss.reset_states()
    per_epoch_mean_acc.reset_states()
    per_epoch_mean_conf.reset_states()
    per_epoch_sample_acc.reset_states()
    per_epoch_sample_conf.reset_states()


    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(data):
        
        
        # Reset batch metrics
        per_batch_loss.reset_states()
        per_batch_mean_acc.reset_states()
        per_batch_mean_conf.reset_states()
        per_batch_sample_acc.reset_states()
        per_batch_sample_conf.reset_states()         
              
                
        # # If the data are batched, flatten the first 2 dimensions
        # if(tf.rank(yc)==4):
        #     xc = flatten_first_two_dims(xc)
        #     yc = flatten_first_two_dims(yc)
        #     xt = flatten_first_two_dims(xt)
        #     yt = flatten_first_two_dims(yt)
        
        # Transpose the y data so they go into the model
        if(tf.rank(yc)==4):
            yc_t = B.transpose(yc,perm=[0,1,3,2])
            yt_t = B.transpose(yt,perm=[0,1,3,2])
        elif(tf.rank(yc)==3):
            yc_t = B.transpose(yc,perm=[0,2,1])
            yt_t = B.transpose(yt,perm=[0,2,1])
        
        
        # print(xc.shape)
        # print(yc.shape)
        # print(xt.shape)
        # print(yt.shape)
        #continue
        
        #pdb.set_trace()
        
        

        # with tf.GradientTape() as tape:
        #     # Compute the loss value for this minibatch.
        #     state = B.global_random_state(B.dtype(xc))
        #     loss = -tf.reduce_mean(
        #         np_elbo_explicit(
        #             state=state,
        #             model=model,
        #             contexts=[(xc,yc_t)],
        #             subsume_context=True,
        #             xt=xt,
        #             yt=yt_t,
        #             normalise=False,
        #             dtype_lik=tf.float32,
        #             num_samples=num_samples,
        #             ))
        with tf.GradientTape() as tape:
            # Compute the loss value for this minibatch.
            state = B.global_random_state(B.dtype(xc))
            loss = -tf.reduce_mean(
                np_elbo_tf_cat(
                    state=state,
                    model=model,
                    contexts=[(xc,yc_t)],
                    subsume_context=True,
                    xt=xt,
                    yt=yt_t,
                    normalise=False,
                    dtype_lik=tf.float32,
                    num_samples=num_samples,
                    ))

        # On the 0th epoch, do not train the model just run metrics for
        # untrained model. On other epochs, calculate and apply the gradients
        if epoch> 0 :
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
       

        # Update loss metrics
        per_epoch_loss(loss)
        per_batch_loss(loss)
        
        
        # Assess accuracy after updating model gradients
        mean, var, noiseless_samples, noisy_samples = nps.predict(
            model,xc, yc_t, xt, num_samples=num_samples
            )
                
        # Redimension data so they are the right dimensions
        mean = tf.transpose(mean, perm=[0, 2, 1])  
        yt_tiled = tf.tile(tf.expand_dims(yt, axis=0), [num_samples, 1, 1, 1])        
        noisy_samples = tf.transpose(noisy_samples, perm=[0,1,3,2])
        
        # Assess accuracy of mean model predictions
        batch_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(yt, mean))
        per_epoch_mean_acc(batch_accuracy)
        per_batch_mean_acc(batch_accuracy)
        
        # Assess accuracy of sampled model predictions
        sample_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(yt_tiled, noisy_samples))
        per_epoch_sample_acc(sample_accuracy)
        per_batch_sample_acc(sample_accuracy)
        
        # Keep track of confidence in model predictions
        per_epoch_mean_conf(calc_cat_confidence(mean, -1))
        per_batch_mean_conf(calc_cat_confidence(mean, -1))
        per_epoch_sample_conf(calc_cat_confidence(noisy_samples, -1))
        per_batch_sample_conf(calc_cat_confidence(noisy_samples, -1))

        
        # End of the batch:
        # Append to batch metrics
        batches_loss.append(per_batch_loss.result())
        batches_mean_acc.append(per_batch_mean_acc.result())
        batches_mean_conf.append(per_batch_mean_conf.result())
        batches_sample_acc.append(per_batch_sample_acc.result())
        batches_sample_conf.append(per_batch_sample_conf.result())
            
        # Keep track of the tasks that have been run
        if not batches_list:
            batches_list.append(0)
        else:
            batches_list.append(batches_list[-1]+1)


    # End of the epoch
    # Keep track of the epochs that have been run
    if not epochs_list:
        epochs_list.append(0)
    else:
        epochs_list.append(epochs_list[-1]+1)
    
    # Append to epoch metrics
    epochs_loss.append(per_epoch_loss.result())
    epochs_mean_acc.append(per_epoch_mean_acc.result())
    epochs_mean_conf.append(per_epoch_mean_conf.result())
    epochs_sample_acc.append(per_epoch_sample_acc.result())
    epochs_sample_conf.append(per_epoch_sample_conf.result())
    
    # Print epoch metrics
    print(f"Elbo Loss: {np.round(float(epochs_loss[-1]),5)}")
    print(f"Accuracy of mean predictions: {np.round(float(epochs_mean_acc[-1]),3)}")
    print(f"Confidence of mean predictions: {np.round(float(epochs_mean_conf[-1]),3)}")
    print(f"Mean accuracy of sampled predictions: {np.round(float(epochs_sample_acc[-1]),3)}")
    print(f"Mean confidence of sampled predictions: {np.round(float(epochs_sample_conf[-1]),3)}")
        
    
# %% Plot training metrics
# Make output folders
figures_path = args.fig

# Plots per epoch

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

# Plotting the data
ax[0].plot(epochs_list, epochs_loss, label='Loss')
ax[1].plot(epochs_list, epochs_mean_acc, label='Mean Accuracy')
ax[1].plot(epochs_list, epochs_mean_conf, label='Mean Confidence')
ax[1].plot(epochs_list, epochs_sample_acc, label='Sample Accuracy')
ax[1].plot(epochs_list, epochs_sample_conf, label='Sample Confidence')

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

# Save the figure
fig.savefig(figures_path+'ex2_training_epochs.png')



# %%


# Fix the task list so 0 is start of first training epoch
tasks_list = [4*element for element in batches_list]

if warmup_epoch:
    tasks_list = [item - 192 for item in tasks_list]

plt.rcParams.update({'font.size': 14}) # set global font size

# Plots per task
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

# Plotting the data
ax[0].plot(tasks_list, batches_loss, label='Loss')
ax[1].plot(tasks_list, batches_mean_acc, label='Mean Accuracy')
ax[1].plot(tasks_list, batches_mean_conf, label='Mean Confidence')
ax[1].plot(tasks_list, batches_sample_acc, label='Sample Accuracy')
ax[1].plot(tasks_list, batches_sample_conf, label='Sample Confidence')

# Adding labels and title
ax[0].set_xlabel('Tasks seen')
ax[0].set_ylabel('Loss')
ax[1].set_xlabel('Tasks seen')
ax[1].set_ylabel('Values')

# Adding legend
ax[0].legend()
ax[1].legend()

ymin0, ymax0 = ax[0].get_ylim()
ymin1, ymax1 = ax[1].get_ylim()

# Add a dotted vertical line
ax[0].vlines(x=-0.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=-0.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=767.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=767.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=1535.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=1535.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=2304.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=2304.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=3071.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=3071.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')



# Add text annotations
ax[0].text(x=(180-192), y=0.5*ymax0, s='Untrained\n model', ha='right')
ax[0].text(x=16, y=0.5*ymax0, s='Epoch 1', ha='left')
#ax[1].text(x=(180-192), y=0.95*ymax1, s='Untrained model', ha='right')
#ax[1].text(x=3, y=0.95*ymax1, s='Training epoch 1', ha='left')
ax[0].text(x=772, y=0.5*ymax0, s='Epoch 2', ha='left')
ax[0].text(x=1552, y=0.5*ymax0, s='Epoch 3', ha='left')
#ax[1].text(x=387, y=0.95*ymax1, s='Training epoch 2', ha='left')
ax[0].text(x=2320, y=0.5*ymax0, s='Epoch 4', ha='left')
#ax[1].text(x=580, y=0.95*ymax1, s='Training epoch 3', ha='left')
ax[0].text(x=3088, y=0.5*ymax0, s='Epoch 5', ha='left')


# Set x tick locations
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(base=768))
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(base=768))


# Make loss y axis logscale
ax[0].set_yscale('symlog')

# Display the plot when running in Spyder
plt.tight_layout()
plt.show()

# Save the figure
fig.savefig(figures_path+'ex2_training_tasks.png')
