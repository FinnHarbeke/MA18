"""
This file is for training the CNN
"""

import torch
from net import FingeralphabetNet
from preprocess import FingeralphabetDataset
import os
from torch.utils.data import DataLoader

# where to save your new Nets
save_path = 'Nets/100epochs/'
# whether or not to train an existing NeuralNet
load = True
# which FingeralphabetNet to train further
load_path = "Nets/100epochs/13.pth"

# initialize CNN
cuda0 = torch.device('cuda:0')
nn = FingeralphabetNet().to(device=cuda0)

# modify the CNN if needed
if load:
    nn.load_state_dict(torch.load(load_path))

abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
dataloader = DataLoader(FingeralphabetDataset('../dataset/train/train'), batch_size=32, shuffle=True, num_workers=4)
if not load:
    torch.save(nn.state_dict(), save_path + '0')
for epoch in range(13, 100):
    nn.train(dataloader, every_batch=100, save_path=save_path + str(epoch + 1) +'.pth')
    #print(*preprocess(5), sep='\n\n')