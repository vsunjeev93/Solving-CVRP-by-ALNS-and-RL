import torch
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, num_hidden, num_out):
        super().__init__()
        # MLP with consistent dimensions
        self.linear1 = torch.nn.Linear(num_hidden, num_hidden)
        self.linear2 = torch.nn.Linear(num_hidden, num_hidden)
        self.linear3 = torch.nn.Linear(num_hidden, num_out)
        
        # Create MLP for GIN layer
        self.MLP1 = torch.nn.Sequential(
            self.linear1,
            torch.nn.ReLU(),
            self.linear2,
            torch.nn.ReLU()
        )
        self.conv1 = GINConv(self.MLP1)
        
    def forward(self, x, data):
        x = self.conv1(x, data.edge_index)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x)
        return x
