import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import random as rnd

# def get_input(filename):
#     """
#     process .png file into pixel tensor
#         filename: path of .png file
#     """
#     img = Image.open(filename)
#     pxls = []
#     #img.show()
#     for rgbInd in range(3):
#         pxls.append([])
#         for j in range(img.size[1]):
#             pxls[rgbInd].append([img.getpixel((i, j))[rgbInd] / 255 for i in range(img.size[0])])
#     return torch.tensor(pxls).float()

# def preprocess(data_path, every=None):
#     """
#     preprocessing for training
#         count: how many to preprocess
#         every: how often to print progress
#     """

#     inputs = []
#     targets = []

#     if type(data_path) is list:
#         iterator = data_path
#     elif type(data_path) is str:
#         iterator = os.listdir(data_path)
#     for step, fn in enumerate(iterator):
#         if fn == '.DS_Store':
#             continue

#         if type(data_path) is list:
#             input_ = get_input(fn)
#         elif type(data_path) is str:
#             input_ = get_input(data_path + '/' + fn)
#         inputs.append(input_)
    
#         abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
#         if fn[1].isdigit():
#             ind = abc.index(fn[0])
#         else:
#             if fn[:3] == "SCH":
#                 ind = 26
#             elif fn[:2] == "CH":
#                 ind = 27
#             else:
#                 ind = 28
#         target = torch.tensor([1 if i == ind else 0 for i in range(29)]).float()
#         targets.append(target)
#         #print(input_.size(), target.size())

#         print(step, 'preprocessed') if every and step % every == 0 else None

#     print('preprocessing done!')
#     return torch.stack(inputs), torch.stack(targets)

# def ImageGenerator(dir, every=2000):
#     for i, fn in enumerate(os.listdir(dir)):
#         if i % every == 0:
#             print('{} Images generated!'.format(i))
#         if fn[-4:] == '.jpg' or fn[-4:] == '.png':
#             yield (Image.open(os.path.join(dir, fn)), fn)

# def tensorBatch(ImgGen, every=2000):
#     X = []
#     y = []
#     for i, (img, fn) in enumerate(ImgGen):
#         if i % every == 0:
#             print('{} Tensors generated!'.format(i))
#         # INPUT
#         X.append(torch.tensor(img.getdata()).transpose(0, 1).view(-1, *img.size[::-1]))
#         # TARGET
#         abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
#         if fn[1].isdigit():
#             ind = abc.index(fn[0])
#         else:
#             if fn[:3] == "SCH":
#                 ind = 26
#             elif fn[:2] == "CH":
#                 ind = 27
#             else:
#                 ind = 28
#         y.append(ind)
#     return torch.stack(X).float() / 255, torch.tensor(y).long()

class FingeralphabetDataset(Dataset):
    def __init__(self, root_dir):
        self.image_names = os.listdir(root_dir)
        if '.DS_Store' in self.image_names:
            self.image_names.remove('.DS_Store')
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, ind):
        fn = self.image_names[ind]
        img = Image.open(os.path.join(self.root_dir, fn))
        image_tensor = torch.tensor(img.getdata()).transpose(0, 1).view(-1, *img.size[::-1])
        image_tensor = image_tensor.float() / 255
        abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
        if fn[1].isdigit():
            target_ind = abc.index(fn[0])
        else:
            if fn[:3] == "SCH":
                target_ind = 26
            elif fn[:2] == "CH":
                target_ind = 27
            else:
                target_ind = 28

        return {'image_tensor': image_tensor, 'target_ind': target_ind}
