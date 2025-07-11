import torch
checkpoint = torch.load('weights/debug_test.pt')
print("Keys in checkpoint:", checkpoint.keys())