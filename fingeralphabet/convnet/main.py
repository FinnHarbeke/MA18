"""
This file is for training the CNN
"""

import torch
from net import Net
from preprocess import preprocess
import os

# where to save your new Nets
save_path = 'Nets/100epochs/'
# whether or not to train an existing NeuralNet
load = False
# which Net to train further
load_path = 'Nets/50all'

# initialize CNN
nn = Net()

# modify the CNN if needed
if load:
    nn.load_state_dict(torch.load(load_path))

abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
for i in range(0, 2700, 1000):
    l = []
    for let in abc:
        jmax = 700 if i == 2000 else 1000
        for j in range(jmax):
            l.append('../dataset/train/' + let + str(i+j) + '.jpg')
    save = save_path+str(i//1000)+'/'
    nn.train(*preprocess(l, every=1000), 100, every=1000, save_path=save)
    #print(*preprocess(5), sep='\n\n')