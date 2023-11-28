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



# Initialise model
print("Initializing model...")

model = anp_ex2(
    dim_embedding=128,
    num_encoder_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
)

#Wessel's suggestion
model2 = nps.construct_agnp(
    dim_x=17,
    dim_y=9,
    num_enc_layers=3,
    num_dec_layers=6,
    dim_embedding=128,
    num_heads=8,
    enc_same=False,        # This is an optimisation you could try. Try setting it to `True`.
    width=512,
    nonlinearity="LeakyReLU",
    likelihood="lowrank",  # Make joint Gaussian predictions!
    num_basis_functions=512,
)


# #Wessel's softmax version
# num_categories = 5

# agnp = nps.construct_agnp(
#     dim_x=2,  # Dimensionality of context and target inputs
#     dim_yc=3,  # Dimensionality of context outputs
#     dim_yt=num_categories,
#     likelihood="het",
#     nonlinearity="leakyrelu",
# )
# agnp.decoder = nps.Chain(
#     # Strip off the heterogeneous likelihood.
#     *agnp.decoder[:-2],  
#     # Add in a softmax.
#     lambda x: torch.softmax(x[..., :num_categories], dim=-1),
# )

# out = agnp(
#     torch.randn(4, 2, 15),  # Context inputs
#     torch.randn(4, 3, 15),  # Context outputs
#     torch.randn(4, 2, 10),  # Target inputs
# )
# print(out.shape)  # Log-probabilities of shape `(4, 5, 10)`

# %%

# Load the data
experiment2_training_hdf = "data/ex2/experiment2_training_data.hdf"

# %%
import pdb
# Assume you have defined your model and optimizer
gnp = nps.construct_gnp(dim_x=17, dim_y=9, dim_lv=0, dim_embedding=2,
                        num_enc_layers=3,num_dec_layers=10,
                        width=16,
                        likelihood="lowrank",
                        enc_same=True) #This improves training if true

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

# Load your data
data = hdf_to_tf_dataset(experiment2_training_hdf)
minibatch_size = 4
data = data.padded_batch(minibatch_size)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Start of epoch {epoch}")

    # Iterate over the batches of the dataset.
    for step, (xc, yc, xt, yt) in enumerate(data):
        #pdb.set_trace()
        
        # Fix the dimensions
        yc = tf.transpose(yc,perm=[0,1,3,2])
        yt = tf.transpose(yt,perm=[0,1,3,2])
        
        with tf.GradientTape() as tape:
            # Compute the loss value for this minibatch.
            loss = -tf.reduce_mean(nps.loglik(gnp, xc,
                                   yc, xt, yt, normalise=True))

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, gnp.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, gnp.trainable_weights))

        # Log every 200 batches.
        if step % 5 == 0:
            print(
                f"Training loss (for one batch) at step {step}: {float(loss)} "
                f"Seen so far: {(step + 1) * minibatch_size} samples"
            )







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