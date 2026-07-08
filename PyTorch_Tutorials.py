import torch
import numpy as np
# Creating an uninitialized tensor (values are undefined):
x = torch.empty(2,2,3)
print(x)
print("----------------------------------------")
# Creating a tensor with random values between 0 and 1:
x = torch.rand(2,2)
print(x)
print("----------------------------------------")
# Creating a tensor filled with zeros:
x = torch.zeros(2,2)
print(x)
print("----------------------------------------")
# Creating a tensor filled with ones:
x = torch.ones(2,2)
print(x)
print("----------------------------------------")
# Displaying the tensor's data type (dtype):
x = torch.ones(2,2)
print(x.dtype)
print("----------------------------------------")
# Creating a tensor with a specified data type:
x = torch.ones(2,2, dtype= torch.int)
print(x.dtype)
print("----------------------------------------")
# Displaying the tensor's shape (dimensions):
x = torch.ones(2,2, dtype= torch.float16)
print(x.size())
print("----------------------------------------")
# Creating a tensor from Python data (a list):
x = torch.tensor([2.5,0.1])
print(x)
print("----------------------------------------")
# Addition:
x = torch.ones(2,2)
y = torch.ones(2,2)
z = x + y   # Equivalent to torch.add(x, y)
print(z)
print("----------------------------------------")
# In-place addition (modifies y directly):
x = torch.ones(2,2)
y = torch.ones(2,2)
y.add_(x)
print(y)
print("----------------------------------------")
# Subtraction:
x = torch.ones(2,2)
y = torch.ones(2,2)
z = x - y   # Equivalent to torch.sub(x, y)
print(z)
print("----------------------------------------")
# In-place subtraction (modifies y directly):
x = torch.ones(2,2)
y = torch.ones(2,2)
y.sub_(x)
print(y)
print("----------------------------------------")
# Multiplication:
x = torch.ones(2,2)
y = torch.ones(2,2)
z = x * y   # Equivalent to torch.mul(x, y)
print(z)
print("----------------------------------------")
# In-place multiplication (modifies y directly):
x = torch.ones(2,2)
y = torch.ones(2,2)
y.mul_(x)
print(y)
print("----------------------------------------")
# Matrix multiplication:
x = torch.ones(2,2)
y = torch.ones(2,2)
z = x @ y   # Equivalent to torch.matmul(x,y)
print(z)
print("----------------------------------------")
# Division:
x = torch.ones(2,2)
y = torch.ones(2,2)
z = x / y   # Equivalent to torch.div(x, y)
print(z)
print("----------------------------------------")
# In-place division (modifies y directly):
x = torch.ones(2,2)
y = torch.ones(2,2)
y.div_(x)
print(y)
print("----------------------------------------")
# Slicing a column:
x = torch.rand(5,3)
print(x)
print(x[:,0])
print("----------------------------------------")
# Slicing a row:
x = torch.rand(5,3)
print(x)
print(x[1,:])
print("----------------------------------------")
# Accessing a specific element:
x = torch.rand(5,3)
print(x)
print(x[1,1])
print("----------------------------------------")
# Getting the Python value of a single-element tensor:
x = torch.rand(5,3)
print(x)
print(x[1,1].item())
print("----------------------------------------")
# Reshaping the tensor without copying data (requires compatible memory layout):
x = torch.rand(4,4)
print(x)
y = x.view(2,8)
print(y)
print("----------------------------------------")
# Reshaping the tensor (returns a view if possible, otherwise returns a copy):
x = torch.rand(4,4)
print(x)
y = x.reshape(2,8)
print(y)
print("----------------------------------------")
# Converting a CPU tensor to a NumPy array (shares memory with the tensor):
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print("----------------------------------------")
# Converting a NumPy array to a tensor (shares memory with the array):
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
print("----------------------------------------")
# Using CUDA (GPU) for tensor operations:
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Method 1: Creating a tensor directly on the GPU
    x = torch.ones(5, device= device)
    # Method 2: Creating a tensor on the CPU and then moving it to the GPU
    y = torch.ones(5)
    y = y.to(device)
    print(x.device)
    print(y.device)
else:
    print("CUDA is not available.")
print("----------------------------------------")
# Converting a GPU tensor to a NumPy array (via the CPU):
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device= device)
    y = torch.ones(5, device= device)
    z = x + y
    # NumPy works only with CPU tensors
    z = z.to("cpu")
    z = z.numpy()
else:
    print("CUDA is not available.")
print("----------------------------------------")
# Creating a tensor with gradient tracking enabled:
x = torch.ones(5, requires_grad= True)
print(x)
print("----------------------------------------")
# Creating a computational graph with gradient tracking:
x = torch.randn(3, requires_grad= True)
print(x)
y = x + 2
print(y)
z = y * y * 2
print(z)
t = z.mean()
print(t)
print("----------------------------------------")
# Computing gradients using backpropagation:
x = torch.randn(3, requires_grad= True)
y = x + 2
z = y * y * 2
z = z.mean()
z.backward()   # Backpropagates through the computational graph and computes gradients
print(x.grad)   # Displaying the computed gradients
print("----------------------------------------")
# Computing gradients for a non-scalar output using a gradient vector:
x = torch.randn(3, requires_grad= True)
y = x + 2
z = y * y * 2
v = torch.tensor([0.1,1.0,0.001], dtype= torch.float32)   # Gradient vector with the same shape as z
z.backward(v)   # Backpropagation with an explicitly provided gradient
print(x.grad)
print("----------------------------------------")
# Disabling gradient tracking for an existing tensor:
x = torch.randn(3, requires_grad= True)
print(x)
x.requires_grad_(False)
print(x)
print("----------------------------------------")
# Creating a tensor detached from the computational graph:
x = torch.randn(3, requires_grad= True)
print(x)
y = x.detach()
print(y)
print("----------------------------------------")
# Temporarily disabling gradient tracking within a code block:
x = torch.randn(3, requires_grad= True)
print(x)
with torch.no_grad():
    y = x + 2
    print(y)
print("----------------------------------------")
# Resetting gradients after each iteration because gradients accumulate by default in PyTorch:
weights = torch.ones(4, requires_grad= True)
for epoch in range(3):
    model_output = (weights+3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()   # Resetting gradients to zero
