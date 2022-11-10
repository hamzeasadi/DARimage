import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
# import torchmetrics as tm
import wandb

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_step(net: nn.Module, data: DataLoader, loss_fn: nn.Module, opt: optim.Optimizer, wbf: bool=False):
    epoch_loss = 0.0
    net.train()
    i = 0

    for X, Y in data:
        i += 1
        out = net(X['src1'].to(dev), X['trg'].to(dev), X['src2'].to(dev))
        loss = loss_fn(Y=Y, pred=out)
        opt.zero_grad()
        loss.backward()
        epoch_loss += (1/i)*(loss.item() - epoch_loss)
        # if (i%100 == 0) and wbf:
        #     wandb.log(dict(train=epoch_loss))

    return dict(epoch_train_loss=epoch_loss)


def eval_step(net: nn.Module, data: DataLoader, loss_fn: nn.Module, wbf: bool=False):
    epoch_loss = 0.0
    net.eval()
    i = 0

    with torch.no_grad():
        for X, Y in data:
            i += 1
            out = net(X['src1'].to(dev), X['trg'].to(dev), X['src2'].to(dev))
            loss = loss_fn(Y=Y, pred=out)
            epoch_loss += (1/i)*(loss.item() - epoch_loss)
            # if (i%100 == 0) and wbf:
            #     wandb.log(dict(val=epoch_loss))

    return dict(epoch_eval_loss=epoch_loss)



def main():
    pass


if __name__ == '__main__':
    main()