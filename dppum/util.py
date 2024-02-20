"""
Several additional definitions for working with NP-based models.
These come from the DP user modelling Julia code
"""

import neuralprocesses as nps
import lab as B

def calc_cat_acc_onehot(y_true,y_pred,cat_axis=-1):
    """
    Calculate the categorical accuracy of one-hot encoded predictions and true
    labels using linear algebra backend.

    Parameters
    ----------
    y_true : array-like
        True labels, one-hot encoded.
    y_pred : array-like
        Predicted labels, one-hot encoded.
    cat_axis : int, optional
        The axis that represents categories, by default -1.

    Returns
    -------
    float
        The categorical accuracy of the predictions.

    Raises
    ------
    ValueError
        If y_true and y_pred are not valid data types for the linear algebra
        backend (lab) library.
        If y_true and y_pred do not have the same shape.
    """
    
    try:
        B.dtype(y_true)
        B.dtype(y_pred)
    except:
        raise ValueError("y_true and y_pred must both be valid data types for the linear algebra backend (lab) library.")
    
    if B.shape(y_true) != B.shape(y_pred):
        raise ValueError("y_true and y_pred do not have the same shape.")
    
    y_true_cat = B.argmax(y_true,cat_axis)
    y_pred_cat = B.argmax(y_pred,cat_axis)
    
    accuracy = B.sum(B.where(B.eq(y_true_cat,y_pred_cat),1,0)) / B.length(y_true_cat)
    
    return accuracy


def calc_cat_confidence(y_pred_onehot, cat_axis):
    """
    Calculate the mean confidence of the most likely prediction of a categorical.

    1. Apply a softmax
    2. Extract the confidence of the most likely category (i.e. the highest value)
    3. Take the mean of all these values

    Parameters
    ----------
    y_pred_onehot : array_like
        One-hot encoded predicted values.
    cat_axis : int
        The categorical axis.

    Returns
    -------
    float
        The mean confidence of the most likely prediction.

    """
    # Normalise y_pred_onehot with a softmax
    y_pred_onehot = B.softmax(y_pred_onehot,axis=cat_axis)
    
    # Calculate confidence of y_pred
    y_pred_confidence = B.max(y_pred_onehot, axis=cat_axis)

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



def print_dictionary(dictionary):
    """
    Print all key-value pairs in a dictionary, line by line.
    
    Parameters
    ----------
    dictionary : dict
        The dictionary to print


    Returns
    -------
    None.

    """
    print("{")
    for key, value in dictionary.items():
        print(f"{key}: {value}")
    print("}")