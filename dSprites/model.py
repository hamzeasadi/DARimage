import torch
from torch import nn as nn
from torchvision import models
from torchinfo import summary

class SprintTNN(nn.Module):
    """
    doc
    """
    def __init__(self) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        base = models.efficientnet_b0(weights=weights)
        self.fx = base.features
        self.adafx = base.avgpool

        for param in self.fx.parameters():
            param.requires_grad = False

        for param in self.fx[7].parameters():
            param.requires_grad = True
        
        for param in self.fx[8].parameters():
            param.requires_grad = True
        
        self.reg = nn.Sequential(
            nn.Linear(in_features=2*1280, out_features=2)
        )

    def forward_once(self, x):
        return self.adafx(self.fx(x))


    def forward(self, x1, x2, x3):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        z3 = self.forward_once(x3)

        z12 = torch.hstack((z1.squeeze(), z2.squeeze()))
        z23 = torch.hstack((z2.squeeze(), z3.squeeze()))
        z31 = torch.hstack((z3.squeeze(), z1.squeeze()))

        y12 = self.reg(z12)
        y23 = self.reg(z23)
        y31 = self.reg(z31)

        return dict(y12=y12, y23=y23, y31=y31, z1=z1.squeeze(), z2=z2.squeeze(), z3=z3.squeeze())



def main():
    x = torch.randn(size=(3, 5))
    u, s, v = torch.svd(x)
    u1, s1, v1 = torch.linalg.svd(x)
    print(u, s, v)
    print(u1, s1, v1)

    model = SprintTNN()
    # summary(model, input_size=[(1, 3, 240, 240), (1, 3, 240, 240), (1, 3, 240, 240)])

    x1 = torch.randn(5, 3, 240, 240)
    x2 = torch.randn(5, 3, 240, 240)
    x3 = torch.randn(5, 3, 240, 240)
    out = model(x1, x2, x3)
    print(out['z1'].shape, out['y12'].shape)
    u, s, v = torch.svd(out['z1'])
    print(u.shape, s.shape, v.shape)

    

if __name__ == '__main__':
    main()