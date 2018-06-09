import torch
from PIL import Image
import os


def get_input(filename):
    img = Image.open(filename)
    pxls = []
    #img.show()
    for j in range(img.size[1]):
        pxls.append([1 if img.getpixel((i, j)) != 0 else 0 for i in range(img.size[0])])
    return torch.tensor(pxls).float()

def preprocess(count, every=None):

    inputs = []
    targets = []

    for i, fn in enumerate(os.listdir('../genLetts_py/data')[:count]):

        input_ = get_input('../genLetts_py/data/' + fn)
        # 1 channel
        input_ = input_.unsqueeze(0)
        inputs.append(input_)

        ind = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(fn[0])
        target = torch.tensor([1 if i == ind else 0 for i in range(26)]).float()
        targets.append(target)
        #print(input_.size(), target.size())

        print(i, 'preprocessed') if every and i % every == 0 else None

    print('preprocessing done!')
    return torch.stack(inputs), torch.stack(targets)

if __name__ == '__main__':

    inp = get_input('../genLetts_py/data/A20.png')

    for row in inp:
        print(*[int(element) for element in row], sep=' ')