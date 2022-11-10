import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
import torchmetrics as tm

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_step(net: nn.Module, data: DataLoader, loss_fn: nn.Module, opt: optim.Optimizer):
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

    return dict(train_loss=epoch_loss)


def eval_step(net: nn.Module, data: DataLoader, loss_fn: nn.Module, opt: optim.Optimizer):
    epoch_loss = 0.0
    net.eval()
    i = 0

    with torch.no_grad():
        for X, Y in data:
            i += 1
            out = net(X['src1'].to(dev), X['trg'].to(dev), X['src2'].to(dev))
            loss = loss_fn(Y=Y, pred=out)
            opt.zero_grad()
            loss.backward()
            epoch_loss += (1/i)*(loss.item() - epoch_loss)

    return dict(eval_loss=epoch_loss)



