#Load local copy of neuralprocesses
import sys
sys.path.insert(0,'/Users/user/github/neuralprocesses')
sys.path.insert(0,'/Users/user/github/stheno')

import neuralprocesses.tensorflow as nps
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
from src.architectures.anp_ex2 import anp_ex2
from src.data import hdf_to_tf_dataset
import pdb
import lab as B
import numpy as np
import matplotlib.pyplot as plt
# %%

# First, parse any command line arguments

# Creating the ArgumentParser instance
parser = argparse.ArgumentParser()

# Adding arguments to the parser
parser.add_argument("--gen", 
                    help="Experiment setting: gridworld, menu_search, h_menu_search", 
                    type=str, 
                    default="menu_search")

parser.add_argument("--n_traj", 
                    help="Number of context trajectories. Setting to 0 randomizes between 1 and 8.", 
                    type=int, 
                    default=0)

parser.add_argument("--batch_size", 
                    help="Batch size.", 
                    type=int, 
                    default=4)

parser.add_argument("--params", 
                    help="Return params?", 
                    type=bool, 
                    default=False)

parser.add_argument("--p_bias", 
                    help="Probability of generating a sample with biased model", 
                    type=float, 
                    default=0.0)

parser.add_argument("--bson", 
                    help="Directly specify the file to save the model to and load it from.", 
                    type=str)

# Parsing the arguments
args = parser.parse_args()


# %%
# Initialise model
# print("Initializing model...")

# model = anp_ex2(
#     dim_embedding=128,
#     num_encoder_heads=8,
#     num_encoder_layers=6,
#     num_decoder_layers=6,
# )

# #Wessel's suggestion
# model2 = nps.construct_agnp(
#     dim_x=17,
#     dim_y=9,
#     num_enc_layers=3,
#     num_dec_layers=6,
#     dim_embedding=128,
#     num_heads=8,
#     enc_same=False,        # This is an optimisation you could try. Try setting it to `True`.
#     width=512,
#     nonlinearity="LeakyReLU",
#     likelihood="lowrank",  # Make joint Gaussian predictions!
#     num_basis_functions=512,
# )

# %%
# #Wessel's corrected softmax version
import neuralprocesses.torch as nps
import torch
 
num_categories = 133
 
agnp = nps.construct_agnp(
    dim_x=2,  # Dimensionality of context and target inputs
    dim_yc=3,  # Dimensionality of context outputs
    dim_yt=num_categories,
    likelihood="het",
    nonlinearity="leakyrelu",
)
agnp.decoder = nps.Chain(
    # Strip off the heterogeneous likelihood.
    *agnp.decoder[:-2], 
    # Add in a softmax.
    lambda x: torch.softmax(x[..., :num_categories, :], dim=-2),
)
 
out = agnp(
    torch.randn(4, 2, 15),  # Context inputs
    torch.randn(4, 3, 15),  # Context outputs
    torch.randn(4, 2, 10),  # Target inputs
)
print(out.shape)  # Log-probabilities of shape `(4, 133, 10)`

# The last dimension should be the number of target points, and the
# second-to-last dim. the number of categories.


# %%

# Load the data
experiment2_training_hdf = "data/ex2/experiment2_training_data.hdf"


# %%
# Calculate the likelihood of the most likely prediction of a categorical
def calc_cat_confidence(y_pred_onehot, categorical_dim):
    # Normalise y_pred_onehot with a softmax
    y_pred_onehot = tf.nn.softmax(y_pred_onehot)
    
    # Calculate confidence of y_pred
    y_pred_confidence = tf.reduce_max(y_pred_onehot, axis=categorical_dim)

    # Calculate the mean confidence
    mean_confidence = tf.reduce_mean(y_pred_confidence)
    
    return mean_confidence

# %%
K.clear_session()
# This version works for standard ELBO loss, but floats in and floats out

# dim_yc = (1,) * 9
# dim_yt = 9


agnp = nps.construct_agnp(dim_x=17, dim_y=9, dim_lv=10, dim_embedding=16,
                        num_enc_layers=6,num_dec_layers=6,
                        likelihood="het",
                        #transform="positive",#doesnt work in training, gives nans
                        enc_same=False) #This improves training if true

print("Running with dim_embedding = 16 for speed but it should be 128")

# agnp = nps.construct_agnp(dim_x=17, dim_yc=dim_yc,dim_yt=dim_yt,
#                           dim_lv=2, dim_embedding=128,
#                         num_enc_layers=6,num_dec_layers=6,
#                         likelihood="het",
#                         #transform="positive",#doesnt work in training, gives nans
#                         enc_same=False) #This improves training if true


gnp = nps.construct_gnp(dim_x=17, dim_y=9, dim_lv=0, dim_embedding=2,
                        num_enc_layers=6,num_dec_layers=6,
                        likelihood="het",
                        enc_same=False) #This improves training if true

model = agnp


# # Set all the model weights to zero initially
# for var in model.trainable_variables:
#     var.assign(tf.zeros_like(var))




optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)

# Load your data
data = hdf_to_tf_dataset(experiment2_training_hdf,dtype=tf.float32)
num_tasks = len(list(data))
#data = data.padded_batch()

#For keeping track of how many tasks the model has seen
num_tasks_seen = 0

# Number of samples taken to assess sample accuracy/confidence
n_test_draws=10

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
per_task_loss = tf.keras.metrics.Mean(name='elbo_loss')
per_task_mean_acc = tf.keras.metrics.Mean(name='mean_acc')
per_task_mean_conf = tf.keras.metrics.Mean(name='mean_confidence')
per_task_sample_acc = tf.keras.metrics.Mean(name='sample_accuracy')
per_task_sample_conf = tf.keras.metrics.Mean(name='sample_conf')

# Keep track of training metrics per task
tasks_list = []
tasks_loss = []
tasks_mean_acc = []
tasks_mean_conf = []
tasks_sample_acc = []
tasks_sample_conf = []


#Testing standard loss
# This trains the model but is wrong?
#loss_fn = tf.keras.losses.CategoricalCrossentropy()

# %%
# Training loop
num_epochs = 2
for epoch in range(num_epochs+1):
    print(f"""######## Start of epoch {epoch} ########""")
    
    # Shuffle the dataset
    #data = data.shuffle(buffer_size=num_tasks)
    
    # Reset epoch metrics
    per_epoch_loss.reset_states()
    per_epoch_mean_acc.reset_states()
    per_epoch_mean_conf.reset_states()
    per_epoch_sample_acc.reset_states()
    per_epoch_sample_conf.reset_states()    


    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(data):
        
        if epoch == 0:
            continue
        
        # if epoch==1:
        #     pdb.set_trace()
        
        # Reset task metrics
        per_task_loss.reset_states()
        per_task_mean_acc.reset_states()
        per_task_mean_conf.reset_states()
        per_task_sample_acc.reset_states()
        per_task_sample_conf.reset_states()         
        
        
        # Fix the dimensions of the y data
        #yc_t = tf.transpose(yc,perm=[0,1,3,2])
        #yt_t = tf.transpose(yt,perm=[0,1,3,2])
        
        #Without padded batch
        yc_t = tf.transpose(yc,perm=[0,2,1])
        yt_t = tf.transpose(yt,perm=[0,2,1])
        
        
        with tf.GradientTape() as tape:
            
            
            # Assess accuracy
            mean, var, noiseless_samples, noisy_samples = nps.predict(
                model,xc, yc_t, xt, num_samples=n_test_draws
                )
            #loss = loss_fn(yt_t, mean)
            
            mean_t = tf.transpose(mean, perm=[0, 2, 1])
            
            probabilities = tf.nn.softmax(mean_t)
            
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(yt,mean_t)
                )
            
            # # Compute the loss value for this minibatch.
            # loss = -tf.reduce_mean(nps.elbo(model, xc,
            #                         yc_t, xt, yt_t, normalise=False,
            #                         dtype_lik=tf.float32,
            #                         num_samples=n_test_draws
            #                         ))

        # with tf.GradientTape() as tape:
        #     # Compute the loss value for this minibatch.
        #     state = B.global_random_state(B.dtype(xc))
        #     loss = -tf.reduce_mean(
        #         elbo2(#elbo_tf_cat(
        #             state=state,
        #             model=agnp,
        #             contexts=[(xc,yc)],
        #             xt=xt,
        #             yt=yt,
        #             normalise=False,
        #             dtype_lik=tf.float32
        #             ))

        # On the 0th epoch, do not train the model just run metrics for
        # untrained model
        if epoch> 0 :
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, model.trainable_weights)
    
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
            # Add to num_tasks
            num_tasks_seen = num_tasks_seen + 1        

        # Keep track of epoch metrics
        per_epoch_loss(loss)
        per_task_loss(loss)
        
        #pdb.set_trace()

        # Assess accuracy after updating model gradients
        mean, var, noiseless_samples, noisy_samples = nps.predict(
            model,xc, yc_t, xt, num_samples=n_test_draws
            )
        # with no padded batch
        mean = tf.transpose(mean, perm=[0, 2, 1])
        #mean = tf.transpose(mean, perm=[0, 1, 3, 2])
        
        yt_reshaped = tf.reshape(yt, [-1, 9])
        mean_reshaped = tf.reshape(mean, [-1, 9])
        
        batch_accuracy = tf.keras.metrics.categorical_accuracy(yt_reshaped, mean_reshaped)
        per_epoch_mean_acc(batch_accuracy)
        per_task_mean_acc(batch_accuracy)
        
        # Keep track of confidence in mean predictions
        # The categorical dimension is now 3 after it was transposed
        # with no padded batch
        per_epoch_mean_conf(calc_cat_confidence(mean, 2))
        per_task_mean_conf(calc_cat_confidence(mean, 2))
        
        #per_epoch_mean_conf(calc_cat_confidence(mean, 3))
        #per_task_mean_conf(calc_cat_confidence(mean, 3))
        
        
        # Loop through sampled predictions and assess accuracy
        for sample in noisy_samples:
            #pdb.set_trace()
            #with no padded batch
            sample = tf.transpose(sample, perm=[0, 2, 1])
            #sample = tf.transpose(sample, perm=[0, 1, 3, 2])
            sample_reshaped = tf.reshape(sample, [-1, 9])
            sample_accuracy = tf.keras.metrics.categorical_accuracy(yt_reshaped, sample_reshaped)
            per_epoch_sample_acc(sample_accuracy)
            per_task_sample_acc(sample_accuracy)
            # with no padded batch
            per_epoch_sample_conf(calc_cat_confidence(sample, 2))
            per_task_sample_conf(calc_cat_confidence(sample, 2))
            #per_epoch_sample_conf(calc_cat_confidence(sample, 3))
            #per_task_sample_conf(calc_cat_confidence(sample, 3))

        
        # Keep track of the metrics
        tasks_loss.append(per_task_loss.result())
        tasks_mean_acc.append(per_task_mean_acc.result())
        tasks_mean_conf.append(per_task_mean_conf.result())
        tasks_sample_acc.append(per_task_sample_acc.result())
        tasks_sample_conf.append(per_task_sample_conf.result())
            
        # End of the task
        # Keep track of the tasks that have been run
        if not tasks_list:
            tasks_list.append(0)
        else:
            tasks_list.append(tasks_list[-1]+1)

    # End of the epoch
    # Keep track of the epochs that have been run
    if not epochs_list:
        epochs_list.append(0)
    else:
        epochs_list.append(epochs_list[-1]+1)
    
    # Keep track of the metrics
    epochs_loss.append(per_epoch_loss.result())
    epochs_mean_acc.append(per_epoch_mean_acc.result())
    epochs_mean_conf.append(per_epoch_mean_conf.result())
    epochs_sample_acc.append(per_epoch_sample_acc.result())
    epochs_sample_conf.append(per_epoch_sample_conf.result())
    
    # Print the metrics
    print(f"Elbo Loss: {np.round(float(epochs_loss[-1]),5)}")
    print(f"Accuracy of mean predictions: {np.round(float(epochs_mean_acc[-1]),3)}")
    print(f"Confidence of mean predictions: {np.round(float(epochs_mean_conf[-1]),3)}")
    print(f"Mean accuracy of sampled predictions: {np.round(float(epochs_sample_acc[-1]),3)}")
    print(f"Mean confidence of sampled predictions: {np.round(float(epochs_sample_conf[-1]),3)}")
    print(f"Seen so far: {num_tasks_seen} tasks")
        
    
    
# Fix the task list so 0 is start of first training epoch
tasks_list = [item - 192 for item in tasks_list]
# %% Plot training metrics

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

# Displaying the plot
plt.tight_layout()
plt.show()

# %%
plt.rcParams.update({'font.size': 14}) # set global font size

# Plots per task
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

# Plotting the data
ax[0].plot(tasks_list, tasks_loss, label='Loss')
ax[1].plot(tasks_list, tasks_mean_acc, label='Mean Accuracy')
ax[1].plot(tasks_list, tasks_mean_conf, label='Mean Confidence')
ax[1].plot(tasks_list, tasks_sample_acc, label='Sample Accuracy')
ax[1].plot(tasks_list, tasks_sample_conf, label='Sample Confidence')

# Adding labels and title
ax[0].set_xlabel('Tasks completed')
ax[0].set_ylabel('Loss')
ax[1].set_xlabel('Tasks completed')
ax[1].set_ylabel('Values')

# Adding legend
ax[0].legend()
ax[1].legend()

ymin0, ymax0 = ax[0].get_ylim()
ymin1, ymax1 = ax[1].get_ylim()

# Add a dotted vertical line at x=191.5
ax[0].vlines(x=-0.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=-0.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=191.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=191.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=383.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=383.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=575.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=575.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')
ax[0].vlines(x=767.5, ymin=ymin0,ymax=ymax0,linestyle='dotted')
ax[1].vlines(x=767.5, ymin=ymin1,ymax=ymax1,linestyle='dotted')



# Add text annotations
ax[0].text(x=(180-192), y=0.95*ymax0, s='Untrained model', ha='right')
ax[0].text(x=3, y=0.95*ymax0, s='Epoch 1', ha='left')
#ax[1].text(x=(180-192), y=0.95*ymax1, s='Untrained model', ha='right')
#ax[1].text(x=3, y=0.95*ymax1, s='Training epoch 1', ha='left')
ax[0].text(x=195, y=0.95*ymax1, s='Epoch 2', ha='left')
ax[0].text(x=387, y=0.95*ymax1, s='Epoch 3', ha='left')
#ax[1].text(x=387, y=0.95*ymax1, s='Training epoch 2', ha='left')
ax[0].text(x=580, y=0.95*ymax1, s='Epoch 4', ha='left')
#ax[1].text(x=580, y=0.95*ymax1, s='Training epoch 3', ha='left')
ax[0].text(x=772, y=0.95*ymax1, s='Epoch 4', ha='left')





# Displaying the plot
plt.tight_layout()
plt.show()

# %% Now try to test trained model

# def my_tf_round(x, decimals=0):
#     multiplier = tf.constant(10**decimals, dtype=x.dtype)
#     return tf.round(x * multiplier) / multiplier

# def calculate_accuracy(y_true_onehot, y_pred_onehot, categorical_dim):
#     pdb.set_trace()
#     # Use argmax to find the index of the maximum value in each one-hot encoded vector
#     y_true_indices = tf.argmax(y_true_onehot, axis=categorical_dim)
#     y_pred_indices = tf.argmax(y_pred_onehot, axis=categorical_dim)

#     # Compare the indices to find the total number of correct predictions
#     correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_true_indices, y_pred_indices), dtype=tf.float32))

#     # Calculate accuracy
#     accuracy = correct_predictions / tf.size(y_true_indices, out_type=tf.float32)
    
#     # Calculate confidence of y_pred
#     y_pred_confidence = tf.reduce_max(y_pred_onehot, axis=categorical_dim)

#     # Calculate the mean confidence
#     mean_confidence = tf.reduce_mean(y_pred_confidence)
    
#     return accuracy, mean_confidence

# def calc_cat_acc(y_true_onehot, y_pred_onehot, categorical_dim):
#     # Use argmax to find the index of the maximum value in each one-hot encoded vector
#     y_true_indices = tf.argmax(y_true_onehot, axis=categorical_dim)
#     y_pred_indices = tf.argmax(y_pred_onehot, axis=categorical_dim)

#     # Compare the indices to find the total number of correct predictions
#     correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_true_indices, y_pred_indices), dtype=tf.float32))

#     # Calculate accuracy
#     accuracy = correct_predictions / tf.size(y_true_indices, out_type=tf.float32)
    
#     return accuracy





# %%
for step, (xc, yc, xt, yt) in enumerate(data_unshuffled):

    pdb.set_trace()
    # Fix the dimensions of the y data
    yc2 = tf.transpose(yc,perm=[0,1,3,2])
    
    out = model(xc,yc2,xt)
    # Use softmax to normalise mean output to probabilities
    probabilities = tf.nn.softmax(tf.transpose(out.mean,perm=[0,1,3,2]))
    
    acc = calculate_accuracy(yt,probabilities,2)
    
    
    


 
    
    
    
    
    
    
    
    
    
    
# %% Trying standard with Wessel's softmax
    
num_categories = 9


agnp = nps.construct_agnp(
    dim_x=17,  # Dimensionality of context and target inputs
    dim_yc=9,  # Dimensionality of context outputs
    dim_yt=num_categories,
    likelihood="het",
    nonlinearity="leakyrelu",
    dim_embedding=128, # From the Julia code
    num_enc_layers=6, # From the Julia code
    num_dec_layers=6, # From the Julia code
    num_heads = 8, # From the Julia code
    dim_lv = 2, #Default is 0 in python- need to check this? Dimensionality of the latent variable.
    
    
)
# agnp.decoder = nps.Chain(
#     # Strip off the heterogeneous likelihood.
#     *agnp.decoder[:-2], 
#     # Add in a softmax.
#     #lambda x: torch.softmax(x[..., :num_categories, :], dim=-1),
#     # Not sure what the comma at the end is for but it was in Wessel's example
#     # I suspect it does nothing
#     lambda x: tf.nn.softmax(x[..., :num_categories, :], axis=-1), 
# )


model = agnp
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)

# Load your data
data = hdf_to_tf_dataset(experiment2_training_hdf,dtype=tf.float32)

#For keeping track of how many tasks the model has seen
num_tasks = len(list(data))

# Set up minibatches
minibatch_size = 5
data = data.padded_batch(minibatch_size)



# %%
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Start of epoch {epoch}")

    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(data):
        #pdb.set_trace()
        
        # Fix the dimensions of the y data
        # The last dimension is the number of target points
        # The second-to-last dimension is the number of categories.
        yc = tf.transpose(yc,perm=[0,1,3,2])
        yt = tf.transpose(yt,perm=[0,1,3,2])
        
        
        with tf.GradientTape() as tape:
            # Compute the loss value for this minibatch.
            loss = -tf.reduce_mean(nps.elbo(model, xc,
                                   yc, xt, yt, normalise=False))
            
            #loss = -nps.elbo(agnp, xc,
            #                       yc, xt, yt2, normalise=False)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)



        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 5 batches.
        if step % 5 == 0:
            print(
                f"Training loss (for one minibatch) at step {step}: {float(loss)} "
                f"Seen so far: {epoch*num_tasks + (step + 1) * minibatch_size} samples"
            )






# %% Now try to test trained model
for step, (xc, yc, xt, yt) in enumerate(data):

    pdb.set_trace()
    # Fix the dimensions of the y data
    yc2 = tf.transpose(yc,perm=[0,1,3,2])
    
    out = model(xc,yc2,xt)
    probabilities = tf.nn.softmax(tf.transpose(out.mean,perm=[0,1,3,2]))

















    

# %%

# Trying to build a custom ELBO loss


# def elbo(labels, logits, mean, logvar):
#     # Log-Likelihood
#     log_likelihood = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

#     # KL Divergence
#     kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)

#     # ELBO is the sum of the expected log-likelihood and the KL divergence
#     return tf.reduce_mean(log_likelihood - kl_divergence)

from neuralprocesses.model.elbo import _merge_context_target, _kl

def elbo_tf_cat(
    state: B.RandomState,
    model: nps.Model,
    contexts: list,
    xt,
    yt,
    *,
    num_samples=1,
    normalise=False,
    subsume_context=False,
    fix_noise=None,
    dtype_lik=None,
    **kw_args,
):
    """ELBO objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.
        dtype_lik (dtype, optional): Data type to use for the likelihood computation.
            Defaults to the 64-bit variant of the data type of `yt`.

    Returns:
        random state, optional: Random state.
        tensor: ELBOs.
    """
    #pdb.set_trace()
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    # For the likelihood computation, default to using a 64-bit version of the data
    # type of `yt`.
    if not dtype_lik:
        dtype_lik = float64

    if subsume_context:
        # Only here also update the targets.
        contexts_q, xt, yt = _merge_context_target(contexts, xt, yt)
    else:
        contexts_q, _, _ = _merge_context_target(contexts, xt, yt)

    # Construct prior.
    xz, pz, h = nps.code_track(
        model.encoder,
        *nps.util.compress_contexts(contexts),
        xt,
        root=True,
        dtype_lik=dtype_lik,
        **kw_args,
    )

    # Construct posterior.
    qz = nps.recode_stochastic(
        model.encoder,
        pz,
        *nps.util.compress_contexts(contexts_q),
        h,
        root=True,
        dtype_lik=dtype_lik,
        **kw_args,
    )

    # Sample from posterior.
    shape = () if num_samples is None else (num_samples,)
    state, z = nps.util.sample(state, qz, *shape)
    z = B.cast(float, z)
    
    # Get the layer before softmax
    pre_softmax_layer = model.layers[-2]

    # Run sample through decoder.
    _, d = nps.code(
        model.decoder,
        xz,
        z,
        xt,
        dtype_lik=dtype_lik,
        root=True,
        **kw_args,
    )
    d = nps.util.fix_noise(d, fix_noise)

    pdb.set_trace()

    # Log-Likelihood
    
    # Transpose y_true and y_pred to shape [minibatch, task_length, num_data_points, num_categories]
    # So they can go into softmax_cross_entropy_with_logits correctly 
    yt_true_transposed = tf.transpose(yt, perm=[0, 1, 3, 2])
    yt_pred_transposed = tf.transpose(d.mean[0], perm=[0, 1, 3, 2])
    #yt_true_transposed = B.cast(dtype_lik,yt_true_transposed)

    # Reconstruction loss
    recon_loss = -tf.nn.softmax_cross_entropy_with_logits(labels=yt_true_transposed, logits=yt_pred_transposed)
    recon_loss = tf.reduce_mean(recon_loss,axis=[-1])

    #kl_term = _kl(qz, pz)
    
    # Compute the ELBO.
    elbos = recon_loss - _kl(qz, pz)
    

    # Compute the ELBO.
    #elbos = B.mean(d.logpdf(B.cast(dtype_lik, yt)), axis=0) - _kl(qz, pz)

    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.cast(dtype_lik, num_data(xt, yt))

#    return state, elbos
    return elbos



def elbo2(
    state: B.RandomState,
    model: nps.Model,
    contexts: list,
    xt,
    yt,
    *,
    num_samples=1,
    normalise=False,
    subsume_context=False,
    fix_noise=None,
    dtype_lik=None,
    **kw_args,
):
    """ELBO objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.
        dtype_lik (dtype, optional): Data type to use for the likelihood computation.
            Defaults to the 64-bit variant of the data type of `yt`.

    Returns:
        random state, optional: Random state.
        tensor: ELBOs.
    """
    #import pdb
    #pdb.set_trace()
    
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    # For the likelihood computation, default to using a 64-bit version of the data
    # type of `yt`.
    if not dtype_lik:
        dtype_lik = float64

    if subsume_context:
        # Only here also update the targets.
        contexts_q, xt, yt = _merge_context_target(contexts, xt, yt)
    else:
        contexts_q, _, _ = _merge_context_target(contexts, xt, yt)

    # Construct prior.
    xz, pz, h = nps.code_track(
        model.encoder,
        *nps.util.compress_contexts(contexts),
        xt,
        root=True,
        dtype_lik=dtype_lik,
        **kw_args,
    )

    # Construct posterior.
    qz = nps.recode_stochastic(
        model.encoder,
        pz,
        *nps.util.compress_contexts(contexts_q),
        h,
        root=True,
        dtype_lik=dtype_lik,
        **kw_args,
    )

    # Sample from posterior.
    shape = () if num_samples is None else (num_samples,)
    state, z = nps.util.sample(state, qz, *shape)
    z = B.cast(float, z)

    # Run sample through decoder.
    _, d = nps.code(
        model.decoder,
        xz,
        z,
        xt,
        dtype_lik=dtype_lik,
        root=True,
        **kw_args,
    )
    d = nps.util.fix_noise(d, fix_noise)
    
    #import pdb
    #pdb.set_trace()
    
    # Compute the ELBO.
    yt2 = B.transpose(yt,[0,1,3,2])
    elbos = B.mean(d.logpdf(B.cast(dtype_lik, yt)), axis=0) - _kl(qz, pz)
    #elbos = B.mean(d.logpdf(B.cast(dtype_lik, yt2)), axis=0) - _kl(qz, pz)
    

    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.cast(dtype_lik, num_data(xt, yt))

    return elbos


# %%

# Test version using custom ELBO loss with categorical data
num_categories = 9


agnp = nps.construct_agnp(
    dim_x=17,  # Dimensionality of context and target inputs
    dim_yc=9,  # Dimensionality of context outputs
    dim_yt=num_categories,
    likelihood="het",
    nonlinearity="leakyrelu",
    dim_embedding=128, # From the Julia code
    num_enc_layers=6, # From the Julia code
    num_dec_layers=6, # From the Julia code
    num_heads = 8, # From the Julia code
    dim_lv = 2, #Default is 0 in python- need to check this? Dimensionality of the latent variable.
    
    
)
agnp.decoder = nps.Chain(
    # Strip off the heterogeneous likelihood.
    *agnp.decoder[:-2], 
    #*agnp.decoder[:], #my test fudge
    # Add in a softmax.
    #lambda x: torch.softmax(x[..., :num_categories, :], dim=-1),
    # Not sure what the comma at the end is for but it was in Wessel's example
    # I suspect it does nothing
    lambda x: tf.nn.softmax(x[..., :num_categories, :], axis=-2), 
)

model = agnp
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-4)

# Load your data
data = hdf_to_tf_dataset(experiment2_training_hdf,dtype=tf.float32)

#For keeping track of how many tasks the model has seen
num_tasks = len(list(data))

# Set up minibatches
minibatch_size = 5
data = data.padded_batch(minibatch_size)




# %%
# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Start of epoch {epoch}")

    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(data):
        
        
        # Fix the dimensions of the y data
        yc = tf.transpose(yc,perm=[0,1,3,2])
        yt = tf.transpose(yt,perm=[0,1,3,2])
        
        #if epoch > 0:
        #    pdb.set_trace()
        
        #pdb.set_trace()
        #xc2 = tf.one_hot(xc,depth=num_categories)
        #yc2 = tf.one_hot(tf.cast(yc, tf.int32),depth=num_categories)
        #xt2 = tf.one_hot(xt,depth=num_categories)
        #yt2 = tf.one_hot(tf.cast(yt, tf.int32),depth=num_categories)
        
        with tf.GradientTape() as tape:
            # Compute the loss value for this minibatch.
            state = B.global_random_state(B.dtype(xc))
            loss = -tf.reduce_mean(
                elbo_tf_cat(
                    state=state,
                    model=agnp,
                    contexts=[(xc,yc)],
                    xt=xt,
                    yt=yt,
                    normalise=False)[1])
            
            #loss = -nps.elbo(agnp, xc,
            #                       yc, xt, yt2, normalise=False)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, agnp.trainable_weights)



        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, agnp.trainable_weights))

        # Log every 5 batches.
        if step % 5 == 0:
            print(
                f"Training loss (for one minibatch) at step {step}: {float(loss)} "
                f"Seen so far: {epoch*num_tasks + (step + 1) * minibatch_size} samples"
            )



# %% Now try to test trained model
for step, (xc, yc, xt, yt) in enumerate(data):

    pdb.set_trace()
    # Fix the dimensions of the y data
    yc = tf.transpose(yc,perm=[0,1,3,2])
    yt = tf.transpose(yt,perm=[0,1,3,2])
    
    out = model(xc,yc,xt)


# %%
# Take output logits from softmax and find the most likely label
def predict_class(labels, logits):
    # Compute the predicted class labels
    predicted_labels = tf.argmax(logits, axis=-1, output_type=tf.int32)
    return predicted_labels



## Initialise loss
#print("Initializing loss...")

# loss() = nps.elbo(
#     xs...,
#     num_samples=5,
#     fixed_σ_epochs=3
# )




#     elbo(
#         model::Model,
#         epoch::Integer,
#         xc::AA,
#         yc::AA,
#         xt::AA,
#         yt::AA;
#         num_samples::Integer,
#         fixed_σ::Float32=1f-2,
#         fixed_σ_epochs::Integer=0,
#         kws...
#     )



# def elbo(
#     model= model,
#     contexts: list,
#     xt,
#     yt,
#     *,
#     num_samples=5,
#     normalise=False,
#     subsume_context=False,
#     fix_noise=None,
#     dtype_lik=None,
#     **kw_args,
    

#%%
#Get the batch size
batch_size  = args["batch_size"]

# Redundant. Required to fit the DataGenerator definition
x_context = nps.UniformContinuous(-2, 2)
x_target  = nps.UniformContinuous(-2, 2)

# Always use 10 context/target points
num_context = nps.UniformDiscrete(lower=10,upper=10)
num_target  = nps.UniformDiscrete(lower=10,upper=10)




data_gen = NeuralProcesses.DataGenerator(
                SearchEnvSampler(args;),
                batch_size=batch_size,
                x_context=x_context,
                x_target=x_target,
                num_context=num_context,
                num_target=num_target,
                σ²=1e-8
            )



# %%

import tensorflow as tf

# Define some integer categorical data
data = tf.constant([1, 2, 3, 4, 5])

# Initialize a CategoryEncoding layer
encoder = tf.keras.layers.CategoryEncoding(
    num_tokens=6,  # Number of unique integers in your data + 1
    output_mode="binary"  # Choose "binary" for one-hot encoding
)


# Initialize an IntegerLookup layer
encoder = tf.keras.layers.IntegerLookup(
    max_tokens=5  # Number of unique integers in your data + 1
)

# Add this line to initialize the tables
tf.tables_initializer().run()

# Call the layer on your data
output = encoder(data)





# Call the layer on your data
output = encoder(data)



#To turn softmax output into integer, use this
# But you can't differentiate it so can't use it in training???
output_integer = tf.argmax(output_softmax, axis=1)
