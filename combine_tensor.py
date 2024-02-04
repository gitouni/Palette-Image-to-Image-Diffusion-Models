import torch

a = torch.load('debug/tensor/train_less_noise.pt',map_location='cpu')
b = torch.load('debug/tensor/train_more_noise.pt',map_location='cpu')
ab = torch.cat([a[None,...],b[None,...]])
torch.save(ab, 'debug/tensor/taxim_m1_train.pt')