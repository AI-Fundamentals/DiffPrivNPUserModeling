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





class MLPCoder:
    """
        MLPCoder

    Code with an MLP. Seems to be some sort of avg pooling in an ANP.
    Based on NeuralProcesses.jl/src/model/coder.jl
    Not 100% clear what exactly it's doing

    # Fields
    - `mlp1`: Pre-pooling MLP.
    - `mlp2`: Post-pooling MLP.
    """

    def __init__(self,mlp1,mlp2):
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def code(self,xz,z,x):
        return x, self.mlp2(B.mean(self.mlp1(B.concat(xz, z, dims=1)), dims=0))
    
    
    
def calc_cat_confidence(y_pred_onehot, categorical_axis):
    """
    Calculate the mean confidence of the most likely prediction of a categorical.

    1. Apply a softmax
    2. Extract the confidence of the most likely category (i.e. the highest value)
    3. Take the mean of all these values

    Parameters
    ----------
    y_pred_onehot : array_like
        One-hot encoded predicted values.
    categorical_axis : int
        The categorical axis.

    Returns
    -------
    float
        The mean confidence of the most likely prediction.

    """
    # Normalise y_pred_onehot with a softmax
    y_pred_onehot = B.softmax(y_pred_onehot,axis=categorical_axis)
    
    # Calculate confidence of y_pred
    y_pred_confidence = B.max(y_pred_onehot, axis=categorical_axis)

    # Calculate the mean confidence
    mean_confidence = B.mean(y_pred_confidence)
    
    return mean_confidence


def flatten_first_two_dims(tensor):
    """
    Flattens the first two dimensions of a tensor.

    Parameters
    ----------
    tensor : Tensor
        The input tensor to be reshaped. The tensor should have at least two
        dimensions.

    Returns
    -------
    Tensor
        The reshaped tensor where the first two dimensions are flattened into
        one, and the remaining dimensions are kept the same.

    """
    
    # Get the shape of the tensor
    shape = B.shape(tensor)
    new_shape = ([shape[0] * shape[1]] + shape[2:]).as_list()

    # Flatten the first two dimensions and keep the remaining dimensions the same
    flattened_tensor = B.reshape(tensor, *new_shape)

    return flattened_tensor


def reshape_to_last(tensor, axis):
    """
    Reshape a tensor so that a specified axis becomes the last dimension.

    Args:
        tensor (B.Tensor): The input tensor.
        axis (int): The axis to move to the last dimension. Can be negative.

    Returns:
        B.Tensor: The reshaped tensor.
    """
    #pdb.set_trace()
    # If the axis is negative, adjust it to be positive
    if axis < 0:
        axis = len(B.shape(tensor)) + axis

    # Get the list of dimensions
    dims = list(range(len(B.shape(tensor))))

    # Remove the specified axis
    dims.remove(axis)

    # Append the specified axis at the end
    dims.append(axis)

    # Use tf.transpose to reshape the tensor
    return B.transpose(tensor, dims)


def logpdf_explicit(d, x, axis=-1):
    """
    Explicitly compute the natural logarithm of the maximum product along each
    on the given axis

    Parameters
    ----------
    d : tensor-like
        The first input tensor.
    x : tensor-like
        The second input tensor.
    axis : int
        The axis along which to compute the calculation. Default is -1.

    Returns
    -------
    tf.Tensor
        A tensor representing the natural logarithm of the maximum product along each row.

    """
    return B.log(B.max(x * d, axis=axis))
