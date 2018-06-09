import torch
from net import Net
from input import preprocess

save_path = '50all'
load = True
load_path = 'fiftythou'


nn = Net()

if load:
    nn.load_state_dict(torch.load(load_path))

nn.train(*preprocess(50000, every=2000), 8, every=2000, save_path=save_path)
#print(*preprocess(5), sep='\n\n')