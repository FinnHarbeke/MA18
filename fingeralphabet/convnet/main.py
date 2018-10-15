"""
This file is for training the CNN
"""

import torch
from net import FingeralphabetNet
from preprocess import FingeralphabetDataset
import os
from torch.utils.data import DataLoader

# where to save your new Nets
save_path = 'Nets/3.try/'
# whether or not to train an existing NeuralNet
load = False
# which FingeralphabetNet to train further
load_path = "Nets/100epochs/88.pth"

# initialize CNN
cuda0 = torch.device('cuda:0')
nn = FingeralphabetNet()
if torch.cuda.is_available():
    nn = nn.to(device=cuda0)

# modify the CNN if needed
if load:
    nn.load_state_dict(torch.load(load_path))

abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
dataloader = DataLoader(FingeralphabetDataset('../dataset/train/train'), 
                batch_size=32, shuffle=True, num_workers=4)
if not load:
    torch.save(nn.state_dict(), save_path + '0.pth')
for epoch in range(30):
    print('Epoch:', epoch+1)
    nn.trainIt(dataloader, every_batch=1000, save_path=save_path + str(epoch + 1) +'.pth')