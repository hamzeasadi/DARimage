import conf as cfg
import secret
import model as m
import engine, utils
import torch
from torch import nn, optim 
from torch.utils.data import DataLoader
import argparse
import wandb
from datetime import datetime
import datasetup as ds


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# arg parse
train_parser = argparse.ArgumentParser(prog='train', description='this parser takes hyperparameters for training, testing and evaluation')

train_parser.add_argument('--train', action=argparse.BooleanOptionalAction)
train_parser.add_argument('--test', action=argparse.BooleanOptionalAction)
train_parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
train_parser.add_argument('--epochs', '-e', metavar='epochs', default=10)
train_parser.add_argument('--lr', type=float, default=9e-4)

args = train_parser.parse_args()

def init_wandb():
    dt = str(datetime.now())
    st = dt.strip().split(' ')[-1].strip().split('.')[0].strip().split(':')
    run_name = '-'.join(st)
    wandb.login(key=secret.wandb_api_key)
    wandb.init(project=f'SpritDA', name=run_name)
    

def train(Net: nn.Module, train_data: DataLoader, val_data: DataLoader, opt: optim.Optimizer, loss_fn: nn.Module, epochs: int, wbf: bool):
    minvalerror = 10e10
    kt = utils.KeepTrack(path=cfg.paths['ckpoint'])
    for epoch in range(epochs):
        train_result = engine.train_step(net=Net, data=train_data, loss_fn=loss_fn, opt=opt, wbf=wbf)
        val_result = engine.eval_step(net=Net, data=val_data, loss_fn=loss_fn, wbf=wbf)

        if val_result['epoch_eval_loss'] < minvalerror:
            minvalerror = val_result['epoch_eval_loss'] 
            kt.save_ckp(model=Net, opt=opt, epoch=epoch, minvalerror=minvalerror, modelName='SprintModelV0.pt')

        wandb.log(train_result|val_result)
        print(train_result|val_result)



def main():
    wandbf = args.wandb
    epochs = args.epochs
    # print(args.lr, epochs)

    model = m.SprintTNN()
    model.to(dev)
    train_l, val_l, test_l = ds.build_dataset(batch_size=64)
    objective = utils.SpritLoss()
    optimizer = utils.build_opt(Net=model, opttype='adam', lr=3e-4)

    if wandbf:
        init_wandb()

    if args.train:
        train(Net=model, train_data=train_l, val_data=val_l, opt=optimizer, loss_fn=objective, epochs=20, wbf=wandbf)
        


if __name__ == '__main__':
    main()