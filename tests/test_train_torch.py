import torch
from dppum.torch.train import get_device_type, average_grads_batch_torch, AverageMeter

def test_get_device_type():
    device = get_device_type()
    assert isinstance(device, str), "Device must be a string"
    
def test_AverageMeter():
    averagemeter = AverageMeter()
    
    averagemeter.update(torch.tensor(2))
    averagemeter.update(torch.tensor(4))
    assert averagemeter.result() == 3
    
    averagemeter.reset()
    assert averagemeter.result() == 0
    
    averagemeter.update(1.5)
    averagemeter.update(torch.tensor(2.5))
    assert averagemeter.result() == 2
    
def average_grads_batch_torch():
    tensor1_1 = torch.ones([10,11])
    tensor1_2 = torch.ones([10,11])*2
    tensor1_3 = torch.ones([10,11])*3
    
    tensor2_1 = torch.ones([5,6])
    tensor2_2 = torch.ones([5,6])*2
    tensor2_3 = torch.ones([5,6])*3
    
    tensors_list = [[tensor1_1,tensor2_1],[tensor1_2,tensor2_2],[tensor1_3,tensor2_3]]
    
    tensors_averaged = average_grads_batch_torch(tensors_list)
    
    assert len(tensors_averaged) == 2
    assert tensors_averaged[0].shape == torch.Size([10, 11])
    assert tensors_averaged[1].shape == torch.Size([5, 6])
    
    
    
    
    