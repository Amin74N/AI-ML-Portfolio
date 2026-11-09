import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1)Data:
class RegressionDataset(Dataset):
    def __init__(self):
        X_NumPy, Y_NumPy = datasets.make_regression(n_samples= 1000, n_features= 1, noise= 20, random_state= 1)
        self.X = torch.from_numpy(X_NumPy.astype(np.float32))
        self.y = torch.from_numpy(Y_NumPy.astype(np.float32)).view(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
# ----------------------------------------------------------------------------------------------------------------------
# 2) Dataset & DataLoader
dataset = RegressionDataset()
data_loader = DataLoader(dataset= dataset, batch_size= 100, shuffle= True)
X = dataset.X
y = dataset.y
# ----------------------------------------------------------------------------------------------------------------------
# 3)Model:
input_size = X.shape[1]
output_size = 1
model = nn.Linear(input_size, output_size)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Loss & Optimizer:
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Training Loop:
num_epochs = 200
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in data_loader:
        # Empty Gradients:
        optimizer.zero_grad()
        # Forward Pass:
        y_predicted = model(batch_X)
        # Loss:
        loss = criterion(y_predicted, batch_y)
        # Backward Pass:
        loss.backward()
        # Update:
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(data_loader)
    # Print Information:
    if (epoch+1) % 5 == 0:
        print(f'epoch: {epoch + 1}/{num_epochs}, loss = {epoch_loss:.4f}')
# ----------------------------------------------------------------------------------------------------------------------
# 5)Plot:
predicted = model(X).detach().numpy()
plt.plot(X.numpy(), y.numpy(), 'ro')
plt.plot(X.numpy()[:, 0], predicted[:, 0], 'b')
plt.show()
