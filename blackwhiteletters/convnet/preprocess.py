import torch
from PIL import Image
import os

def get_input(filename):
    """
    process .png file into pixel tensor
        filename: path of .png file
    """
    img = Image.open(filename)
    pxls = []
    #img.show()
    for j in range(img.size[1]):
        pxls.append([1 if img.getpixel((i, j)) != 0 else 0 for i in range(img.size[0])])
    return torch.tensor(pxls).float()

def preprocess(data_path, every=None):
    """
    preprocessing for training
        count: how many to preprocess
        every: how often to print progress
    """

    inputs = []
    targets = []

    os_list = os.listdir(data_path)
    for i in range(len(os_list)):
        fn = os_list[i]
        if fn == '.DS_Store':
            continue
            
        input_ = get_input(data_path + '/' + fn)
        # 1 channel
        input_ = input_.unsqueeze(0)
        inputs.append(input_)
    
        abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SPACE"]
        ind = abc.index(fn[0]) if fn[:5] != 'SPACE' else 26
        target = torch.tensor([1 if i == ind else 0 for i in range(27)]).float()
        targets.append(target)
        #print(input_.size(), target.size())

        print(i, 'preprocessed') if every and i % every == 0 else None

    print('preprocessing done!')
    return torch.stack(inputs), torch.stack(targets)

if __name__ == '__main__':

    inp = get_input('../genLetts_py/new_data/A20.png')

    for row in inp:
        print(*[int(element) for element in row], sep=' ')