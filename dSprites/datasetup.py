import os
import conf as cfg
import torchvision







def main():
    listc = os.listdir(cfg.paths['dcolor'])
    print(listc[0])



if __name__ == '__main__':
    main()