import torch

# 指定设备为cuda:1
device = torch.device('cuda:1')

# 在cuda:1上创建一个张量
x = torch.tensor([10.0], device=device)

# 执行一个简单的计算
y = x * x

# 打印结果
print(y)
import cupy
print("Cupy version: ", cupy.__version__)
print("CUDA version: ", cupy.cuda.runtime.runtimeGetVersion())
