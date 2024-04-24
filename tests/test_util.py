import numpy as np
import pytest
import torch
import lab as B

# Some tests fail if you don't import neuralprocesses.
# It's something to do with dispatch in Plum?
import neuralprocesses.torch as nps

from dppum.util import (
    calc_cat_acc_onehot,
    calc_cat_confidence,
    flatten_first_two_dims,
    reshape_to_last,
    swap_axes,
    logpdf_explicit
    )


def test_calc_cat_acc_onehot():
    # 1D tensorflow tensors
    y_pred = torch.tensor([0, 0, 1], dtype=torch.float32)
    y_true = torch.tensor([0, 0, 1], dtype=torch.float32)
    assert calc_cat_acc_onehot(y_true,y_pred) == 1.0
    
    # 2D numpy arrays
    y_true = np.array([[0,1,0],[0,0,1]])
    y_pred = np.array([[0,1,0],[1,0,0]])
    assert calc_cat_acc_onehot(y_true,y_pred) == 0.5

    # 3D numpy arrays
    y_true = np.array([
        [[0, 0, 1], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ])

    y_pred =  np.array([
        [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    ])
    
    assert calc_cat_acc_onehot(y_true,y_pred) == pytest.approx(2/3, 0.001)
    
    # Check the cat_axis
    y_true = B.transpose(y_true,perm=[0,2,1])
    y_pred = B.transpose(y_pred,perm=[0,2,1])
    assert calc_cat_acc_onehot(y_true,y_pred,cat_axis=-2) == pytest.approx(2/3, 0.001)
    
    # Add padding
    y_true =  np.array([
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    ])
    y_pred =  np.array([
        [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    ])
    
    padding_value = -1
    assert calc_cat_acc_onehot(y_true,y_pred,cat_axis=-2,padding_value=padding_value) == pytest.approx(0.833, 0.02)
    padding_value = np.array([
        [[False, False, False], [False, False, False], [False, False, False]],
        [[False, False, False], [False, False, False], [False, False, False]],
        [[True, True, True], [True, True, True], [True, True, True]]
    ])
    assert calc_cat_acc_onehot(y_true,y_pred,cat_axis=-2,padding_value=padding_value) == pytest.approx(0.833, 0.02)
    padding_value = np.array([
        [False, False, False],
        [False, False, False],
        [True, True,True]
    ])
    assert calc_cat_acc_onehot(y_true,y_pred,cat_axis=-2,padding_value=padding_value) == pytest.approx(0.833, 0.02)
    
    # Check shapes without averaging
    accuracy_not_averaged = calc_cat_acc_onehot(y_true,y_pred,cat_axis=-1,avg=False)
    assert B.shape(accuracy_not_averaged) == B.shape(y_true)[0:2]
    assert B.mean(accuracy_not_averaged) == calc_cat_acc_onehot(y_true,y_pred,cat_axis=-1,avg=True)
    
    # Check it raises an exception if you try use a list
    with pytest.raises(ValueError):
        calc_cat_acc_onehot([1,0,0],[0,1,0])
                            
    # Check it raises an exception if y_pred and y_true are not the same shapes
    with pytest.raises(ValueError):
        calc_cat_acc_onehot([1,0,0],[1,0])
        
    # Check it raises an exception if padding is the wrong shape
    with pytest.raises(ValueError):
        calc_cat_acc_onehot(y_true,y_pred,-1,np.array([True,False]))
    

def test_calc_cat_confidence():
    # 2D numpy arrays
    logits_2D = np.array([[-100,0.001,100],[99,-0.001,-99]])
    # Axis 0, it should be looking at very high confidence for points 0&2 but low for 1
    assert calc_cat_confidence(logits_2D,0) == pytest.approx(0.833, 0.01)
    # Axis 1, it should be looking at very high confidence for all points
    assert calc_cat_confidence(logits_2D,1) == pytest.approx(1.0, 0.01)
    # Double check default axis
    assert calc_cat_confidence(logits_2D) == pytest.approx(1.0, 0.01)
    
    # Now test with padding_mask
    padding_mask = np.array([[False,False,False],[True,True,True]])
    assert calc_cat_confidence(logits_2D,padding_mask=padding_mask) == pytest.approx(1.0, 0.01)
    padding_mask = B.any(padding_mask,1)
    assert calc_cat_confidence(logits_2D,1,padding_mask=padding_mask) == pytest.approx(1.0, 0.01)
    
    
    
    
def test_flatten_first_two_dims():
    starting_shape = (4,3,2,1)
    data = np.random.rand(*starting_shape)    
    data_flat = flatten_first_two_dims(data)
    assert data_flat.shape == (12,2,1)
    
    
def test_reshape_to_last():
    starting_shape = (4,3,2,1)
    data = np.random.rand(*starting_shape)
    data_reshaped = reshape_to_last(data,1)
    assert data_reshaped.shape == (4,2,1,3)


def test_swap_axes():
    # Check it swaps axes correctly
    starting_shape = (4,3,2,1)
    data = np.random.rand(*starting_shape)
    data_reshaped = swap_axes(data,0,2)
    assert data_reshaped.shape == (2,3,4,1)
    data_reshaped = swap_axes(data,-2,-1)
    assert data_reshaped.shape == (4,3,1,2)
    
    # Check it raises an exception if you specify an invalid axis
    with pytest.raises(IndexError):
        swap_axes(data,0,7)
    
    
    
