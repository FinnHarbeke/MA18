"""
This file is for training the CNN
"""

import torch
import torchvision as tv
from net import FingeralphabetNet
from preprocess import FingeralphabetDataset
import os
from torch.utils.data import DataLoader
os.system("taskset -p 0xff %d" % os.getpid())

# where to save your new Nets
save_path = 'Nets/8.try/'
# whether or not to train an existing NeuralNet
load = False
# which FingeralphabetNet to train further
load_path = "Nets/100epochs/88.pth"

# initialize CNN
nn = FingeralphabetNet()
if torch.cuda.is_available():
    nn = nn.cuda()

# modify the CNN if needed
if load:
    nn.load_state_dict(torch.load(load_path))

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
for epoch in range(20):
    print('Epoch:', epoch+1)
    nn.trainIt(dataloader, every_batch=1000, save_path=save_path + str(epoch + 1) +'.pth')
