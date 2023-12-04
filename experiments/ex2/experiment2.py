#Load local copy of neuralprocesses
import sys
sys.path.insert(0,'/Users/user/github/neuralprocesses')

import neuralprocesses.tensorflow as nps
import argparse
import tensorflow as tf
from src.architectures.anp_ex2 import anp_ex2
from src.data import hdf_to_tf_dataset


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
num_categories = 8

# Make a standard AGNP
agnp = nps.construct_agnp(
    dim_x=17,  # Dimensionality of context and target inputs
    dim_yc=9,  # Dimensionality of context outputs
    dim_yt=num_categories,
    likelihood="het",
    nonlinearity="leakyrelu",
)


# agnp.decoder = nps.Chain(    
#     # Strip off the heterogeneous likelihood.
#     *agnp.decoder[:-2],  
#     # Add in a softmax.
#     lambda x: tf.nn.softmax(x[..., :num_categories], axis=-1)
# )

out = agnp(
    tf.random.uniform([1, 4, 17, 15], minval=0, maxval=num_categories, dtype=tf.float32),  # Context inputs
    tf.random.uniform([1, 4, 9, 15], minval=0, maxval=num_categories, dtype=tf.float32),  # Context outputs
    tf.random.uniform([1, 4, 17, 10], minval=0, maxval=num_categories, dtype=tf.float32),  # Target inputs
)
print(out.shape)  # Log-probabilities of shape `(4, 5, 10)`

# %%

# Load the data
experiment2_training_hdf = "data/ex2/experiment2_training_data.hdf"

# %%

def elbo(labels, logits, mean, logvar):
    # Log-Likelihood
    log_likelihood = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    # KL Divergence
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)

    # ELBO is the sum of the expected log-likelihood and the KL divergence
    return tf.reduce_mean(log_likelihood - kl_divergence)

# %%
def diff_indices_tensor(tensor_list1, tensor_list2):
    # Initialize a list to hold the indices of the differing tensors
    diff_indices = []

    # Compare each pair of tensors
    for i, (tensor1, tensor2) in enumerate(zip(tensor_list1, tensor_list2)):
        if not tf.reduce_all(tf.equal(tensor1, tensor2)):
            diff_indices.append(i)

    return diff_indices

# %%
import pdb
# Assume you have defined your model and optimizer
gnp = nps.construct_gnp(dim_x=17, dim_y=9, dim_lv=2, dim_embedding=128,
                        num_enc_layers=3,num_dec_layers=10,
                        width=16,
                        likelihood="lowrank",
                        enc_same=True) #This improves training if true

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

#This is specified in the supplement of the paper
num_categories = 8

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
    
    

print("Setting up data generator...")
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
