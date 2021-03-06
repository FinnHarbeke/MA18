"""
This file is for training the CNN
"""

import torch
from net import Net
from preprocess import preprocess

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

nn.train(*preprocess('../genLetts_py/new_data', every=2000), 100, every=2000, save_path=save_path)
#print(*preprocess(5), sep='\n\n')