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
    
    