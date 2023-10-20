"""
Several additional definitions for working with NP-based models.
"""

import neuralprocesses as nps
import lab as B



    
    


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





