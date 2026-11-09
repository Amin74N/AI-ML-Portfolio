import torch
import torch.nn as nn
import torch.nn.functional as F
# ----------------------------------------------------------------------------------------------------------------------
class NeuralNetwork(nn.Module):                     # nn.Module inheritance
    def __init__(self):                             # Constructor Function
        super().__init__()
        self.fc1 = nn.Linear(2,16)                  # Layer 1
        self.fc2 = nn.Linear(16,8)                  # Layer 2
        self.fc3 = nn.Linear(8,1)                   # Layer 3
    def forward(self, x):                           # Forward Function
        x = self.fc1(x)
        x = F.relu(x)                               # ReLU Activation
        x = self.fc2(x)
        x = F.relu(x)                               # ReLU Activation
        x = self.fc3(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------
model = NeuralNetwork()                             # Building Multi-Layer Perceptron (MLP) Model
print(model)                                        # Displaying Model Structure
# ----------------------------------------------------------------------------------------------------------------------
sample = torch.tensor([[5.0,90.0]])
prediction = model(sample)                          # Forward Pass for Prediction
print("Random output:", prediction)                 # Predicting Before Training
# ----------------------------------------------------------------------------------------------------------------------
total_params = 0
for p in model.parameters():                        # Calculating Number of Model Parameters
    total_params += p.numel()
print("Number of model parameters:", total_params)  # Displaying Number of Model Parameters
