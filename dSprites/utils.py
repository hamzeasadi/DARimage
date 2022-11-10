import torch
from torch import nn as nn, optim as optim
import os


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpritLoss(nn.Module):
    """
    doc
    """
    def __init__(self) -> None:
        super().__init__()
        self.crt = nn.SmoothL1Loss()

    def RSD(self, Feature_s1, Feature_t, Feature_s2):
        u_s1, s_s1, v_s1 = torch.svd(Feature_s1)
        u_s2, s_s2, v_s2 = torch.svd(Feature_s2)
        u_t, s_t, v_t = torch.svd(Feature_t)
        
        rots1 = torch.mm(u_s1, v_s1.t())
        rots2 = torch.mm(u_s2, v_s2.t())
        rott = torch.mm(u_t, v_t.t())

        lr1 = self.MMD(x=rots1, y=rott, kernel='rbf')
        lr2 = self.MMD(x=rots2, y=rott, kernel='rbf')
       

        return lr1+lr2

    def MMD(self, x, y, kernel):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        
        XX, YY, XY = (torch.zeros(xx.shape, device=dev),
                    torch.zeros(xx.shape, device=dev),
                    torch.zeros(xx.shape, device=dev))
        
        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        if kernel == "rbf":
        
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)    

        return torch.mean(XX + YY - 2. * XY)


    def forward(self, Y: dict, pred: dict):

        y_zero = pred['y12'] + pred['y23'] + pred['y31']
        y1_y3hat = pred['y12'] + pred['y23']
        y1_y3 = Y['y_src1'] - Y['y_src2']
        yhat1 = Y['y_src2'] - pred['y31']
        yhat3 = pred['y31'] + Y['y_src1']

        # print(y1_y3hat.shape)
        # print(y1_y3.shape)
    
        loss1 = self.crt(y_zero, torch.zeros_like(y_zero, device=dev))
        loss2 = self.crt(yhat1.squeeze(), Y['y_src1'].squeeze())
        loss3 = self.crt(yhat3.squeeze(), Y['y_src2'].squeeze())
        loss4 = self.crt(y1_y3hat.squeeze(), y1_y3.squeeze())
        loss5 = self.RSD(Feature_s1=pred['z1'], Feature_s2=pred['z3'], Feature_t=pred['z2'])
        
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        return loss


class KeepTrack():
    """
    doc
    """
    def __init__(self, path: str) -> None:
        self.state = dict(model_state='', opt_state='', min_error=0.0, epoch=0)
        self.path = path
    
    def save_ckp(self, net: nn.Module, opt: optim.Optimizer, epoch: int, min_error: float, model_name: str):
        self.state['model_state'] = net.state_dict()
        self.state['opt_state'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['min_error'] = min_error
        save_path = os.path.join(self.path, model_name)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, model_name: str):
        return torch.load(f=os.path.join(self.path, model_name))


def build_opt(Net: nn.Module, opttype: str='adam', lr: float=9e-4):
    if opttype == 'adam':
        opt = optim.Adam(params=Net.parameters(), lr=lr)
    elif opttype == 'sgd':
        opt = optim.SGD(params=Net.parameters(), lr=lr)

    return opt





def main():
    pass



if __name__ == '__main__':
    main()