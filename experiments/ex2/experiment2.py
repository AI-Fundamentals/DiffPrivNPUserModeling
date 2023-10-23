

import neuralprocesses as nps
from src.architectures.anp_ex2 import anp_ex2


# Initialise model

model = anp_ex2(
    dim_embedding=128,
    num_encoder_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
)
