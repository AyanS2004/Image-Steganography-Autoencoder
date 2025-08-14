import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())  # Usually 0
print(torch.cuda.get_device_name(0))  # Your GPU name
