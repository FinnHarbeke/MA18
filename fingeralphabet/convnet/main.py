"""
This file is for training the CNN
"""

import torch
from net import FingeralphabetNet
from preprocess import FingerAlphabetDataset
import os
from torch.utils.data import DataLoader

# where to save your new Nets
save_path = 'Nets/100epochs/'
# whether or not to train an existing NeuralNet
load = False
# which FingeralphabetNet to train further
load_path = None

# initialize CNN
nn = FingeralphabetNet()

# modify the CNN if needed
if load:
    nn.load_state_dict(torch.load(load_path))

abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
dataloader = DataLoader(FingerAlphabetDataset('../dataset/train'), batch_size=32, shuffle=True)
torch.save(nn.state_dict(), save_path + '0')
for epoch in range(100):
    nn.train(dataloader, every_batch=1000, save_path=save_path + str(epoch + 1))
    #print(*preprocess(5), sep='\n\n')