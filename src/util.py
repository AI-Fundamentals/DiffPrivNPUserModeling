"""
Several additional definitions for working with NP-based models.
These come from the DP user modelling Julia code
"""

import neuralprocesses as nps
import lab as B


def build_categorical_noise(
        build_local_transform=lambda n: n,
        dim_y=1
        ):
    """
    Build a categorical noise model.

    Parameters
    ----------
    build_local_transform : function, optional
        Function to build local transformation (default identity function),
        meaning no local transformation will be used.
    dim_y : int, optional
        Dimensionality of y (default 1).

    Returns
    -------
    tuple
        A tuple containing the number of noise channels and the constructed
        noise model.
    """
    
    num_noise_channels = dim_y
    noise = nps.Chain([
        build_local_transform(dim_y),  # Apply the local transformation
        SoftmaxLikelihood(4)  # Apply a softmax
    ])

    return num_noise_channels, noise




class SoftmaxLikelihood(nps.likelihood.AbstractLikelihood):
    """
    SoftmaxLikelihood class represents a softmax likelihood function.
    Essentially a list of probabilities that sum up to 1.

    Parameters
    ----------
    k : int
        Number of available actions.
    """

    def __init__(self, k=4):
        """
        Initializes the SoftmaxLikelihood class with the provided number of
        available actions.

        Parameters
        ----------
        k : int (Default: 4)
            Number of available actions.
        """
        self.k = k

    def __call__(self, x):
        """
        Computes the action probabilities based on the input tensor by running
        a softmax.

        Parameters
        ----------
        x : tensor
            Input tensor.

        Returns
        -------
        Categorical
            Categorical distribution generated from the computed probabilities.
        """
        p = B.softmax(x, axis=1)
        return Categorical(probs=p)

    
    


class Categorical:
    """
    A class representing a categorical distribution.

    Parameters
    ----------
    p : array-like
        Event probabilities.
    """

    def __init__(self, p):
        """
        Initializes the Categorical distribution with the provided event probabilities.

        Parameters
        ----------
        p : array-like
            Event probabilities.
        """
        self.p = p





