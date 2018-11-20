"""
This file is for training the CNN
"""

import torch
from net import FingeralphabetNet
from preprocess import FingeralphabetDataset
import os
from torch.utils.data import DataLoader
import torchvision as tv

# where to save your new Nets
save_path = 'Nets/9.try/'
# whether or not to train an existing NeuralNet
load = False
# which FingeralphabetNet to train further
load_path = "Nets/100epochs/88.pth"

# initialize CNN
cuda = torch.device('cuda')
nn = FingeralphabetNet()
if torch.cuda.is_available():
    nn = nn.to(device=cuda)

# modify the CNN if needed
if load:
    nn.load_state_dict(torch.load(load_path))

abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
dataset = tv.datasets.ImageFolder('../dataset/train2', transform=tv.transforms.Compose([
    tv.transforms.Resize((200, 150)),
    tv.transforms.RandomCrop((160, 120)),
    tv.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    tv.transforms.RandomRotation(30),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([.5]*3, [.5]*3),
]))
dataloader = DataLoader(dataset, 
                batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
if not load:
    torch.save(nn.state_dict(), save_path + '0.pth')
for epoch in range(30):
    print('Epoch:', epoch+1)
    nn.trainIt(dataloader, every_batch=1000, save_path=save_path + str(epoch + 1) +'.pth')