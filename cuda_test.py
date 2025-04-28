import torch

print("CUDA_count:", torch.cuda.device_count())
print("CUDA_name:", torch.cuda.get_device_name(0))

print(torch.cuda.is_available())
