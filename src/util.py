"""
Several additional definitions for working with NP-based models.
"""

import neuralprocesses as nps
import lab as B






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





