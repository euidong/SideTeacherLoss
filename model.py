import os
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_c=1, in_w=28, in_h=28, out_c=10):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, 3, 3),
            nn.Conv2d(3, 6, 5),
            nn.Flatten(),
            nn.Linear((in_w - 6)*(in_h - 6)*6, 32*32),
            nn.ReLU(),
            nn.Linear(32 * 32, out_c),
        )

    def forward(self, x) -> torch.Tensor:
        output = self.layers(x)
        return output
    
    def save(self, name: str = 'model') -> None:
        # get file list in param directory and get next index
        torch.save(self.state_dict(), f"{name}.pt")

    def load(self, name: str = 'model') -> None:
        try:
            self.load_state_dict(torch.load(f"{name}.pt"))
        except Exception as e:
            print(e)
            print(f"Can't load {name}.pt")