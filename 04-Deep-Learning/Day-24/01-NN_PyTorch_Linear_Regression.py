### Design model (input/output size, forward pass):
import torch
import torch.nn as nn
X = torch.tensor([[1],[2],[3],[4]], dtype= torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype= torch.float32)
X_test = torch.tensor([5], dtype= torch.float32)
# ----------------------------------------------------------- #
# Input/Output size:
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features
# ----------------------------------------------------------- #
# Forward pass:
class MyLinRegMod(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyLinRegMod, self).__init__()
        self.lin = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.lin(x)
model = MyLinRegMod(input_size, output_size)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
# ----------------------------------------------------------------------------------------------------------------------
### Construct loss & optimizer:
learning_rate = 0.1
n_iters = 100
# ----------------------------------------------------------- #
# Loss:
loss = nn.MSELoss()
# ----------------------------------------------------------- #
# Optimizer:
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
# ----------------------------------------------------------------------------------------------------------------------
### Training Loop:
for epoch in range(n_iters):
# ----------------------------------------------------------- #
# Forward pass (compute prediction):
    y_pred = model(X)
    l = loss(y_pred, Y)
# ----------------------------------------------------------- #
# Backward pass (gradients):
    l.backward()
# ----------------------------------------------------------- #
# Update weights:
    optimizer.step()
    optimizer.zero_grad()
# ----------------------------------------------------------- #
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f},b = {b[0].item():.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
