import os


root = os.path.join(os.pardir, 'data')
paths = dict(
    root=root, dsprites=os.path.join(root, 'dSprites'),
    dcolor=os.path.join(root, 'dSprites', 'color'),
    dnoisy=os.path.join(root, 'dSprites', 'noisy'),
    dscream=os.path.join(root, 'dSprites', 'scream'),
    ckpoint=os.path.join(root, 'ckpoint')
)


def main():
    print(paths)


if __name__ == '__main__':
    main()