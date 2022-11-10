import os
import conf as cfg
import torchvision
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import io, transforms
# from torchvision.tra
import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data(path: str):
    data = pd.read_csv(path, header=None, sep=' ').values
    for i, row in enumerate(data):
        data[i][0] = row[0].split('/')[1].strip()
    
    return data


def transformx(auto):
    if auto:
        weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        t = weight.transforms()

    else:
        t = transforms.Compose(
            transforms=[transforms.Resize((224, 224)), transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
    return t

class SpritData(Dataset):
    """
    doc
    """
    def __init__(self, path: dict=cfg.paths) -> None:
        super().__init__()
        self.path = path
        self.color = get_data(os.path.join(self.path['dcolor'], 'color.txt'))
        self.len_ = len(self.color)
        self.transform = transformx(auto=True)
        self.src1 = np.random.choice(a=np.arange(self.len_), size=self.len_, replace=False)
        self.src2 = np.random.choice(a=np.arange(self.len_), size=self.len_, replace=False)
        self.trg = np.random.choice(a=np.arange(self.len_), size=self.len_, replace=False)
        
    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        idx_src1 = self.src1[index]
        idx_src2 = self.src2[index]
        idx_trg = self.trg[index]
        src1_instance = self.color[idx_src1]
        src2_instance = self.color[idx_src2]
        trg_instance = self.color[idx_trg]

        src1img = self.transform(io.read_image(path=os.path.join(self.path['dcolor'], src1_instance[0])))
        src2img = self.transform(io.read_image(path=os.path.join(self.path['dcolor'], src2_instance[0])))
        trgimg = self.transform(io.read_image(path=os.path.join(self.path['dscream'], trg_instance[0])))
        y_src1 = torch.tensor(list(src1_instance[[3, 4]]), device=dev)
        y_src2 = torch.tensor(list(src2_instance[[3, 4]]), device=dev)
        y_trg = torch.tensor(list(trg_instance[[3, 4]]), device=dev)
        return dict(src1=src1img, src2=src2img, trg=trgimg), dict(y_src1=y_src1, y_src2=y_src2, y_trg=y_trg)

def build_dataset(dataset: Dataset=SpritData(), train_percent: float=0.8, batch_size: int=64):
    total_size = len(dataset)
    train_size = int(train_percent*total_size)
    evaluation_size = total_size - train_size
    validation_size = int(evaluation_size*0.8)
    test_size = evaluation_size - validation_size
    train, evaluation = random_split(dataset=dataset, lengths=[train_size, evaluation_size])
    validation, test = random_split(dataset=evaluation, lengths=[validation_size, test_size])
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    
    return train_loader, validation_loader, test_loader


def main():
    # listc = os.listdir(cfg.paths['dcolor'])
    dataset = SpritData()
    train_loader, validation_loader, test_loader = build_dataset(dataset=dataset, batch_size=5)
    X, Y = next(iter(test_loader))
    print(X['src1'].shape)
    print(Y['y_src1'])
    print(Y['y_trg'])


if __name__ == '__main__':
    main()