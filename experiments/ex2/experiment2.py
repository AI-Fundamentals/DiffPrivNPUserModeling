#Load local copy of neuralprocesses
import sys
sys.path.insert(0,'/Users/user/github/neuralprocesses')

import neuralprocesses.tensorflow as nps
import argparse
import tensorflow as tf
from src.architectures.anp_ex2 import anp_ex2
from src.data import hdf_to_tf_dataset
import pdb
import lab as B
import numpy as np
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
# #Wessel's softmax version
num_categories = 4

# Make a standard AGNP
agnp = nps.construct_agnp(
    dim_x=17,  # Dimensionality of context and target inputs
    dim_yc=9,  # Dimensionality of context outputs
    dim_yt=num_categories,
    likelihood="het",
    nonlinearity="leakyrelu",
)


agnp.decoder = nps.Chain(    
    # Strip off the heterogeneous likelihood.
    *agnp.decoder[:-2],
    # Add in a softmax.
    lambda x: tf.nn.softmax(x[..., :num_categories], axis=-1)
)

out = agnp(
    tf.random.uniform([1, 4, 17, 15], minval=0, maxval=num_categories, dtype=tf.float32),  # Context inputs
    tf.random.uniform([1, 4, 9, 15], minval=0, maxval=num_categories, dtype=tf.float32),  # Context outputs
    tf.random.uniform([1, 4, 17, 10], minval=0, maxval=num_categories, dtype=tf.float32),  # Target inputs
)
print(out.shape)  # Log-probabilities of shape `(4, 5, 10)`





# %% #Wessel's Original softmax version
import neuralprocesses.torch as nps
import torch


num_categories = 3

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
    lambda x: torch.softmax(x[..., :num_categories], dim=-1),
)

out = agnp(
    torch.randn(4, 2, 15),  # Context inputs
    torch.randn(4, 3, 15),  # Context outputs
    torch.randn(4, 2, 10),  # Target inputs
)
print(out.shape)  # Log-probabilities of shape `(4, 5, 10)`

# %%

# Load the data
experiment2_training_hdf = "data/ex2/experiment2_training_data.hdf"

# %%

# This version works for standard ELBO loss, but floats in and floats out
agnp = nps.construct_agnp(dim_x=17, dim_y=9, dim_lv=2, dim_embedding=128,
                        num_enc_layers=3,num_dec_layers=10,
                        width=16,
                        likelihood="het",
                        enc_same=False) #This improves training if true


model = agnp


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

# Load your data
data = hdf_to_tf_dataset(experiment2_training_hdf,dtype=tf.float32)

minibatch_size = 4
data = data.padded_batch(minibatch_size)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Start of epoch {epoch}")

    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(data):
        pdb.set_trace()
        
        # Fix the dimensions of the y data
        yc = tf.transpose(yc,perm=[0,1,3,2])
        yt = tf.transpose(yt,perm=[0,1,3,2])
        
        
        
        #xc2 = tf.one_hot(xc,depth=num_categories)
        #yc2 = tf.one_hot(tf.cast(yc, tf.int32),depth=num_categories)
        #xt2 = tf.one_hot(xt,depth=num_categories)
        #yt2 = tf.one_hot(tf.cast(yt, tf.int32),depth=num_categories)
        
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

        # Log every 200 batches.
        if step % 5 == 0:
            print(
                f"Training loss (for one batch) at step {step}: {float(loss)} "
                f"Seen so far: {(step + 1) * minibatch_size} samples"
            )





# %% Now try to test trained model
for step, (xc, yc, xt, yt) in enumerate(data):

    pdb.set_trace()
    # Fix the dimensions of the y data
    yc = tf.transpose(yc,perm=[0,1,3,2])
    yt = tf.transpose(yt,perm=[0,1,3,2])
    
    out = agnp(xc,yc,xt)

# %%
def elbo(labels, logits, mean, logvar):
    # Log-Likelihood
    log_likelihood = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    # KL Divergence
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)

    # ELBO is the sum of the expected log-likelihood and the KL divergence
    return tf.reduce_mean(log_likelihood - kl_divergence)

from neuralprocesses.model.elbo import _merge_context_target

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
    pdb.set_trace()
    
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

    pdb.set_trace()

    # Log-Likelihood
    #log_likelihood = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=d)
    
    # KL Divergence
    #kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
    
    # ELBO is the sum of the expected log-likelihood and the KL divergence
    #return tf.reduce_mean(log_likelihood - kl_divergence)


    # Compute the ELBO.
    elbos = B.mean(d.logpdf(B.cast(dtype_lik, yt)), axis=0) - _kl(qz, pz)

    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.cast(dtype_lik, num_data(xt, yt))

    return state, elbos

# %%

# Test version using custom ELBO loss with categorical data
# agnp = nps.construct_agnp(dim_x=17, dim_y=9, dim_lv=2, dim_embedding=128,
#                         num_enc_layers=3,num_dec_layers=10,
#                         width=16,
#                         likelihood="het",
#                         enc_same=False) #This improves training if true
# #This is specified in the supplement of the paper
# num_categories = 8
# agnp.decoder = nps.Chain(    
#     # Strip off the heterogeneous likelihood.
#     *agnp.decoder[:-2],  
#     # Add in a softmax.
#     lambda x: tf.nn.softmax(x[..., :num_categories], axis=-1)
# )

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

# Load your data
data = hdf_to_tf_dataset(experiment2_training_hdf,dtype=tf.float32)

minibatch_size = 1
data = data.padded_batch(minibatch_size)



# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Start of epoch {epoch}")

    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(data):
        
        
        # Fix the dimensions of the y data
        yc = tf.transpose(yc,perm=[0,1,3,2])
        yt = tf.transpose(yt,perm=[0,1,3,2])
        
        
        
        #xc2 = tf.one_hot(xc,depth=num_categories)
        #yc2 = tf.one_hot(tf.cast(yc, tf.int32),depth=num_categories)
        #xt2 = tf.one_hot(xt,depth=num_categories)
        #yt2 = tf.one_hot(tf.cast(yt, tf.int32),depth=num_categories)
        
        with tf.GradientTape() as tape:
            # Compute the loss value for this minibatch.
            state = B.global_random_state(B.dtype(xc))
            a,loss = -tf.reduce_mean(
                elbo_tf_cat(
                    state=state,
                    model=agnp,
                    contexts=[(xc,yc)],
                    xt=xt,
                    yt=yt,
                    normalise=False))
            
            #loss = -nps.elbo(agnp, xc,
            #                       yc, xt, yt2, normalise=False)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, agnp.trainable_weights)



        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, agnp.trainable_weights))

        # Log every 200 batches.
        if step % 5 == 0:
            print(
                f"Training loss (for one batch) at step {step}: {float(loss)} "
                f"Seen so far: {(step + 1) * minibatch_size} samples"
            )




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
