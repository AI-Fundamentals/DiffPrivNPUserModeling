from dppum.torch.train import get_device_type, average_grads_batch_torch, AverageMeter

def test_get_device_type():
    device = get_device_type()
    assert isinstance(device, str), "Device must be a string"
    
def test_AverageMeter():
    averagemeter = AverageMeter()
    
    averagemeter.update(2)
    averagemeter.update(4)
    
    assert averagemeter.result() == 3
    
    