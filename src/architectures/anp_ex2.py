import neuralprocesses.tensorflow as nps
from src.util import build_categorical_noise, MLPCoder

def anp_ex2(dim_embedding,
            num_encoder_layers,
            num_encoder_heads,
            num_decoder_layers
    ):
    """
    Construct the attentive neural process (ANP) used in experiment 2.
    
    Parameters
    ----------
    dim_embedding : int
        Embedding dimension in the MLPs.
    num_encoder_layers : int
        Number of layers in the encoder MLPs.
    num_encoder_heads : int
        Number of encoder attention heads.
    num_decoder_layers : int
        Number of layers in the decoder MLP.

    Returns
    -------
    neuralprocesses Model
        Attentive neual process model.

    """
    dim_x = 17
    dim_y = 9

    # Build noise channels
    num_noise_channels, noise = build_categorical_noise(dim_y=dim_y)

    # The encoder is made of 3 parts that go in parallel:
    # Part 1: Deterministic part
    encoder_pt1 = nps.Chain(
        nps.InputsCoder(),
        nps.DeterministicLikelihood()
    )

    # Part 2: Attention
    encoder_pt2 = nps.Chain(
        nps.Attention(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            num_heads=num_encoder_heads,
            num_enc_layers=num_encoder_layers
        ),
        nps.DeterministicLikelihood()
    )    

    # Part 3: Encoder MLPs, using MLPCoder for average pooling
    encoder_pt3 = nps.Chain(
        MLPCoder(
            nps.MLP(
                dim_in=dim_x + dim_y,
                dim_hidden=dim_embedding,
                dim_out=dim_embedding,
                num_layers=num_encoder_layers,
                nonlinearity='LeakyReLU'
            ),
            nps.MLP(
                dim_in=dim_embedding,
                dim_hidden=dim_embedding,
                dim_out=2 * dim_embedding,
                num_layers=num_encoder_layers,
                nonlinearity='LeakyReLU'
            )
        ),
        nps.HeterogeneousGaussianLikelihood()
    )

    # Join the 3 encoder parts together in parallel
    encoder = nps.Parallel(encoder_pt1, encoder_pt2, encoder_pt3)

    # The decoder consists mainly of an MLP and some noise
    decoder = nps.Chain(
        nps.Materialise(),
        nps.MLP(
            dim_in=dim_x + 2 * dim_embedding,
            dim_hidden=dim_embedding,
            dim_out=num_noise_channels,
            num_layers=num_decoder_layers,
            nonlinearity='LeakyReLU'
        ),
        noise
    )

    return nps.Model(encoder, decoder)

