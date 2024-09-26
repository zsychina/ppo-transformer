import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        # [batch, state_dim]
        mlp = self.mlp(x)
        return self.head(mlp)


if __name__ == '__main__':
    x = torch.rand(1, 512)
    mlp = MLP(512, 1024, 10)
    out_dist = mlp(x)
    out_dist = F.softmax(out_dist, dim=-1)
    print(out_dist.shape)
    dist = torch.distributions.Categorical(out_dist)
    print(dist.sample())
