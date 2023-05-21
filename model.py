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
    
    def save(self, prefix: str = 'model') -> None:
        # get file list in param directory and get next index
        
        file_list = os.listdir('param')
        max_idx = -1
        for file in file_list:
            try:
                idx = int(file.split(prefix)[1].split('.')[0])
                max_idx = max(max_idx, idx)
            except:
                continue
        idx = max_idx + 1
        torch.save(self.state_dict(), f"param/{prefix}{idx}.pth")

    def load(self, idx: int, prefix: str = 'model') -> None:
        try:
            self.load_state_dict(torch.load(f"param/{prefix}{idx}.pth"))
        except Exception as e:
            print(e)
            print(f"Can't load {prefix}{idx}.pth")