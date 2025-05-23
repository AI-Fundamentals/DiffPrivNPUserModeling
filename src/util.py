# SPDX-FileCopyrightText: 2025 University of Manchester
#
# SPDX-License-Identifier: apache-2.0

"""
Several additional definitions for working with NP-based models.
These come from the DP user modelling Julia code
"""

import lab as B
import torch

def calc_greedy_acc_onehot(y_true,y_pred,cat_axis=-1,padding_value=None,avg=True):
    """
    Calculate the categorical accuracy of one-hot encoded predictions and true
    labels using linear algebra backend.

    Parameters
    ----------
    y_true : array-like
        True labels, one-hot encoded.
    y_pred : array-like
        Predicted labels, soft one-hot encoded.
    cat_axis : int, optional
        The axis that represents categories, by default -1.
    padding_value : float, optional
        Value that represents padding in y_true and y_pred. These values will
        not be used in the averaging.
    avg : bool, optional
        If true this will return one value. If false, the average will be the
        shape of 'y_true' collapsed over 'cat_axis', with the same padding as
        y_true based on padding_value.

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
        
    if padding_value is not None and B.size(padding_value) != 1:
        raise ValueError("padding_value must be a single float.")
        
    # Calculate the categories (i.e. not one-hot)
    y_true_cat = B.argmax(y_true,cat_axis)
    y_pred_cat = B.argmax(y_pred,cat_axis)    
    accuracy = B.cast(B.dtype(y_true),y_true_cat == y_pred_cat)
    # At this stage accuracy is of the shape as y_true, but collapsed over the
    # categorical dimension

    # If there is padding, create the padding mask
    if padding_value is not None:
        padding_mask = (y_true == padding_value)
        padding_mask = B.any(padding_mask, axis=cat_axis)
    
    # Returns
    if avg:
        if padding_value is not None:
            return B.mean(accuracy[~padding_mask])
        else:
            return B.mean(accuracy)            
    else:
        if padding_value is not None:
            # Assign padding values to the padding parts
            accuracy[padding_mask] = padding_value
        return accuracy
    
    
def calc_true_confidence(y_true_onehot,y_pred_onehot, cat_axis=-1, padding_value=None):
    """
    Calculate the mean confidence of the true (i.e. correct) category.

    1. Apply a softmax.
    2. Extract the confidence of the true category (i.e. the highest value).
    3. Take the mean of all these values (if applicable).

    Parameters
    ----------
    y_true : array-like
        True labels, one-hot encoded.
    y_pred : array-like
        Predicted labels, soft one-hot encoded.
    cat_axis : int
        The categorical axis. Default value is -1.
    padding_value : float, optional
        Value that represents padding in y_pred_onehot and y_true. These values
        will not be used in the averaging.

    Returns
    -------
    float
        The mean confidence of the true category.

    """
    
    if padding_value is not None and B.size(padding_value) != 1:
        raise ValueError("padding_value must be a single float.")
        
    if not isinstance(y_pred_onehot,torch.Tensor):
        raise TypeError("y_pred_onehot must be a pytorch tensor")
    if not isinstance(y_true_onehot,torch.Tensor):
        raise TypeError("y_true_onehot must be a pytorch tensor")
    
    # Normalise y_pred_onehot with a softmax
    y_pred_onehot = B.softmax(y_pred_onehot,axis=cat_axis)
    
    # Get the true category
    y_true_cat = B.argmax(y_true_onehot, axis=cat_axis)
    
    # Use the true category to index y_pred_onehot to get the confidence of the
    # true category (need to make y_true_cat of dtype int64).
    y_true_confidence  = torch.gather(y_pred_onehot, dim=cat_axis, index=y_true_cat.int().long().unsqueeze(dim=cat_axis))
    y_true_confidence = y_true_confidence.squeeze()

    # Calculate the mean confidence
    # If there is padding, create the padding mask
    if padding_value is not None:
        padding_mask = (y_true_onehot == padding_value)
        padding_mask = B.any(padding_mask, axis=cat_axis)
        padding_mask = B.squeeze(padding_mask)
        mean_confidence = B.mean(y_true_confidence[~padding_mask])
    else:
        # Calculate the mean confidence, ignoring padding
        mean_confidence = B.mean(y_true_confidence)
    
    return mean_confidence


def calc_greedy_confidence(y_pred_onehot, cat_axis=-1, padding_value=None):
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
        The categorical axis. Default value is -1.
    padding_value : float, optional
        Value that represents padding in y_pred_onehot. These values will not
        be used in the averaging.

    Returns
    -------
    float
        The mean confidence of the most likely prediction.

    """

    # Normalise y_pred_onehot with a softmax
    y_pred_onehot = B.softmax(y_pred_onehot,axis=cat_axis)
    
    # Calculate confidence of y_pred
    y_pred_confidence = B.max(y_pred_onehot, axis=cat_axis)
    
    if padding_value is not None and B.size(padding_value) != 1:
        raise ValueError("padding_value must be a single float.")
    
    # Calculate the mean confidence
    # If there is padding, create the padding mask
    if padding_value is not None:
        padding_mask = (y_pred_onehot == padding_value)
        padding_mask = B.any(padding_mask, axis=cat_axis)            
        mean_confidence = B.mean(y_pred_confidence[~padding_mask])
    else:
        # Calculate the mean confidence, ignoring padding
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
    new_shape = list( (shape[0]*shape[1],) + shape[2:] )

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

    # Reshape the tensor
    return B.transpose(tensor, dims)



def swap_axes(tensor, axis1, axis2):
    """
    Swap two specified axes of a tensor.

    Args:
        tensor (B.Tensor): The input tensor.
        axis1 (int): The first axis to swap. Can be negative.
        axis2 (int): The second axis to swap. Can be negative.

    Returns:
        B.Tensor: The tensor with swapped axes.
    """
    
    if axis1 >= len(tensor.shape) or axis2 >= len(tensor.shape):
        raise IndexError("Axis value is greater than tensor dimensions")
    
    # If the axes are negative, adjust them to be positive
    if axis1 < 0:
        axis1 = len(B.shape(tensor)) + axis1
    if axis2 < 0:
        axis2 = len(B.shape(tensor)) + axis2

    # Get the list of dimensions
    dims = list(range(len(B.shape(tensor))))

    # Swap the specified axes
    dims[axis1], dims[axis2] = dims[axis2], dims[axis1]

    # Reshape the tensor
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
    B.Tensor
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
    


def average_grads_batch_torch(tensor_list):
    """
    Compute the average of tensors over the batch dimension. This is designed
    to average a list of lists of gradients into one single list of gradients.

    Parameters
    ----------
    tensor_list : list of list of torch.Tensor
        The input list of lists of tensors. The dimensions of this list of
        lists are [batch,num_tensors,tensor_sizes].

    Returns
    -------
    list of torch.Tensor
        The output list of averaged tensors. The dimensions of this list are
        [num_tensors,tensor_sizes].

    """
    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the number of tensors
    for i in range(len(tensor_list[0])):
        # Initialize a list to store the tensors of the current index from each batch
        tensors = []

        # Iterate over the batches
        for j in range(len(tensor_list)):
            # Append the tensor of the current index from the current batch to the list
            tensors.append(tensor_list[j][i])

        # Convert the list of tensors to a 3D tensor
        tensors = torch.stack(tensors)

        # Calculate the mean over the batch dimension (0th dimension) and
        # append the result to the averaged tensors list
        averaged_tensors.append(torch.mean(tensors, dim=0))

    # Return the list of averaged tensors
    return averaged_tensors