#Load local copy of neuralprocesses
import sys
sys.path.insert(0,'/Users/user/github/neuralprocesses')

import neuralprocesses as nps
import argparse
from src.architectures.anp_ex2 import anp_ex2


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

## Initialise loss
print("Initializing loss...")

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
x_context = nps.uniform.UniformContinuous(-2, 2)
x_target  = Distributions.Uniform(-2, 2)

# Always use 10 context/target points
num_context = 10#nps.dist.B.randint(lower=10,upper=11)
num_target  = 10#nps.dist.B.randint(lower=10,upper=11)




data_gen = NeuralProcesses.DataGenerator(
                SearchEnvSampler(args;),
                batch_size=batch_size,
                x_context=x_context,
                x_target=x_target,
                num_context=num_context,
                num_target=num_target,
                σ²=1e-8
            )