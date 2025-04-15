import torch
import torch.nn as nn
import torch.optim as optim
from fp4_torch_kernel.layers import FP4Linear
from fp4_torch_kernel.optimizers import FP4Adam
import random, numpy

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)

class SimpleFP4Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleFP4Model, self).__init__()
        self.fc1 = FP4Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features, dtype=torch.bfloat16)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train():
    device = "cuda"
    model = SimpleFP4Model(10, 20, 5).to(device)
    optimizer = FP4Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.randn(4, 10, device=device, dtype=torch.bfloat16)
    y = torch.randn(4, 5, device=device, dtype=torch.bfloat16)

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

if __name__ == '__main__':
    train()
