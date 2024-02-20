import numpy as np
import pytest
import tensorflow as tf
import lab as B

# Some tests fail if you don't import neuralprocesses.
# It's something to do with dispatch in Plum?
import neuralprocesses.tensorflow as nps


from dppum.util import calc_cat_acc_onehot





def test_calc_cat_acc_onehot():
    # 1D tensorflow tensors
    y_pred = tf.convert_to_tensor([0,0,1],dtype=tf.float32)
    y_true = tf.convert_to_tensor([0,0,1],dtype=tf.float32)
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
    

    
    
    # Check it raises an exception if you try use a list
    with pytest.raises(ValueError):
        calc_cat_acc_onehot([1,0,0],[0,1,0])
                            
    # Check it raises an exception if y_pred and y_true are not the same shapes
    with pytest.raises(ValueError):
        calc_cat_acc_onehot([1,0,0],[1,0])
    


