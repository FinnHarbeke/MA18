"""
This file is for training the CNN
"""

import torch
from net import Net
from preprocess import tensorBatch, ImageGenerator
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
nn.train(*tensorBatch(ImageGenerator('../dataset/train')), 100, every=1000, save_path=save)
    #print(*preprocess(5), sep='\n\n')