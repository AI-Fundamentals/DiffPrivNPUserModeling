import numpy as np
import pytest
import torch
import lab as B

# Some tests fail if you don't import neuralprocesses.
# It's something to do with dispatch in Plum?
import neuralprocesses.torch as nps

from src.util import (
    calc_greedy_acc_onehot,
    calc_greedy_confidence,
    calc_true_confidence,
    flatten_first_two_dims,
    reshape_to_last,
    swap_axes,
    logpdf_explicit,
    average_grads_batch_torch
    )


def test_calc_greedy_acc_onehot():
    # 1D torch tensors
    y_pred = torch.tensor([0, 0, 1], dtype=torch.float32)
    y_true = torch.tensor([0, 0, 1], dtype=torch.float32)
    assert calc_greedy_acc_onehot(y_true,y_pred) == 1.0
    
    # 2D numpy arrays
    y_true = np.array([[0,1,0],[0,0,1]])
    y_pred = np.array([[0,1,0],[1,0,0]])
    assert calc_greedy_acc_onehot(y_true,y_pred) == 0.5

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
    
    assert calc_greedy_acc_onehot(y_true,y_pred) == pytest.approx(2/3, 0.001)
    
    # Check the cat_axis
    y_true = B.transpose(y_true,perm=[0,2,1])
    y_pred = B.transpose(y_pred,perm=[0,2,1])
    assert calc_greedy_acc_onehot(y_true,y_pred,cat_axis=-2) == pytest.approx(2/3, 0.001)
    
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
    assert calc_greedy_acc_onehot(y_true,y_pred,cat_axis=-2,padding_value=padding_value) == pytest.approx(0.833, 0.02)

    # Check shapes without averaging
    accuracy_not_averaged = calc_greedy_acc_onehot(y_true,y_pred,cat_axis=-1,avg=False)
    assert B.shape(accuracy_not_averaged) == B.shape(y_true)[0:2]
    assert B.mean(accuracy_not_averaged) == calc_greedy_acc_onehot(y_true,y_pred,cat_axis=-1,avg=True)
    
    # Check values without averaging, with padding
    padding_value = -1.
    accuracy_not_averaged = calc_greedy_acc_onehot(y_true,y_pred,cat_axis=-1,avg=False,padding_value=-1.)
    assert B.shape(accuracy_not_averaged) == B.shape(y_true)[0:2]
    padding_mask = (y_true == padding_value)
    padding_mask = B.any(padding_mask, axis=-1)
    assert B.mean(accuracy_not_averaged[~padding_mask]) == calc_greedy_acc_onehot(y_true,y_pred,cat_axis=-1,avg=True,padding_value=-1.)
    
    # Check it raises an exception if you try use a list
    with pytest.raises(ValueError):
        calc_greedy_acc_onehot([1,0,0],[0,1,0])
                            
    # Check it raises an exception if y_pred and y_true are not the same shapes
    with pytest.raises(ValueError):
        calc_greedy_acc_onehot([1,0,0],[1,0])
        
    # Check it raises an exception if padding is an array
    with pytest.raises(ValueError):
        calc_greedy_acc_onehot(y_true,y_pred,-1,np.array([True,False]))
    

def test_calc_greedy_confidence():
    # 2D numpy arrays
    logits_2D = np.array([[-100,0.001,100],[99,-0.001,-99]])
    # Axis 0, it should be looking at very high confidence for points 0&2 but low for 1
    assert calc_greedy_confidence(logits_2D,0) == pytest.approx(0.833, 0.01)
    # Axis 1, it should be looking at very high confidence for all points
    assert calc_greedy_confidence(logits_2D,1) == pytest.approx(1.0, 0.01)
    # Double check default axis
    assert calc_greedy_confidence(logits_2D) == pytest.approx(1.0, 0.01)
    
    # Check it raises an exception if padding is an array
    with pytest.raises(ValueError):
        padding_mask = np.array([[False,False,False],[True,True,True]])
        assert calc_greedy_confidence(logits_2D,padding_value=padding_mask) == pytest.approx(1.0, 0.01)


def test_calc_true_confidence():

    y_pred_onehot = torch.tensor(
        [[-100,0.001,100],
         [99,-0.001,-99]]
    )
    
    # Axis 0, the actual categories are [1,1,0]
    y_true_onehot = torch.tensor([[0.,0.,1.],
                                  [1.,1.,0.]])
    # So the confidence of these should be [1,~0.5,1], so mean of about 0.833
    assert calc_true_confidence(y_true_onehot,y_pred_onehot,0) == pytest.approx(0.833, 0.01)
    
    # Same but transposed and axis 1
    assert calc_true_confidence(y_true_onehot.T,y_pred_onehot.T,1) == pytest.approx(0.833, 0.01)
    
    # Test padding
    padding_value = -1.
    # No padding in the data
    assert calc_true_confidence(y_true_onehot,y_pred_onehot,0,padding_value) == pytest.approx(0.833, 0.01)
    
    pad = (1,1)
    y_true_onehot_pad = torch.nn.functional.pad(y_true_onehot,pad,"constant",padding_value)
    y_pred_onehot_pad = torch.nn.functional.pad(y_pred_onehot,pad,"constant",padding_value)
    assert calc_true_confidence(y_true_onehot_pad,y_pred_onehot_pad,0,padding_value) == pytest.approx(0.833, 0.01)   
    
    
    # Check it raises an exception if padding is an array
    with pytest.raises(ValueError):
        padding_mask = torch.tensor([[False,False,False],[True,True,True]])
        calc_true_confidence(y_true_onehot,y_pred_onehot,0,padding_value=padding_mask)
        
    # Check it raises an exception if inputs are not pytorch tensors
    with pytest.raises(TypeError):
        numpy_array = np.array([[-100,0.001,100],[99,-0.001,-99]])
        calc_true_confidence(numpy_array,y_pred_onehot,0)
    with pytest.raises(TypeError):
        numpy_array = np.array([[-100,0.001,100],[99,-0.001,-99]])
        calc_true_confidence(y_true_onehot,numpy_array,0)
        

    
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
        
        
def test_logpdf_explicit():
    data1 = torch.tensor([1,2,3,4,5])
    data2 = torch.tensor([5,4,3,2,1])
    assert logpdf_explicit(data1,data2,0) == B.log(9)
    
    data1 = torch.tensor([[1,2,3,4,5],[1,2,3,4,5]])
    data2 = torch.tensor([[5,4,3,2,1],[1,2,3,4,5]])
    assert torch.equal(logpdf_explicit(data1,data2,1),B.log(torch.tensor([9,25])))
    assert torch.equal(logpdf_explicit(data1,data2,0),B.log(torch.tensor([5,8,9,16,25])))
    
    
def test_average_grads_batch_torch():
    shape1 = (3,4)
    shape2 = (4,3)
    shape3 = (5,5)
    # Average of all these should be 2
    list1 = [torch.full(shape1, 1.), torch.full(shape2, 1.), torch.full(shape3, 1.)]
    list2 = [torch.full(shape1, 3.), torch.full(shape2, 3.), torch.full(shape3, 3.)]
    list3 = [torch.full(shape1, 4.), torch.full(shape2, 4.), torch.full(shape3, 4.)]
    list4 = [torch.full(shape1, 0.), torch.full(shape2, 0.), torch.full(shape3, 0.)]
    
    biglist = [list1,list2,list3,list4]
    
    biglist_avg = average_grads_batch_torch(biglist)
    
    avglist_check = [torch.full(shape1, 2.), torch.full(shape2, 2.), torch.full(shape3, 2.)]
    
    
    assert all(torch.allclose(tensor1, tensor2) for tensor1, tensor2 in zip(biglist_avg, avglist_check))


    
    
    
