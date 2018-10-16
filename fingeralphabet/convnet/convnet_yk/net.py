import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class FingeralphabetNet(nn.Module):

    def __init__(self):
        super(FingeralphabetNet, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(  3,  32, 3, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d( 32,  64, 3, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d( 64, 128, 3, padding=1, bias=False)
        self.batch3 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(38400, 29)
        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()
        self.optim = torch.optim.Adam(self.parameters(), 1e-4)
        

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.batch1(x)), 2)
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(self.batch2(x)), 2)
        x = self.conv3(x)
        x = F.max_pool2d(F.relu(self.batch3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def trainIt(self, dataloader, every_batch=2000, save_path=None):
        """
        train function using a torch.utils.data.DataLoader()
            :param dataloader: torch.utils.data.DataLoader()
            :param every_batch=2000: after how many batches to report error etc.
            :param save_path=None: where to save the net, if None FingeralphabetNet is not savec during training
        """ 
        self.train(True)
        running_error = 0
        for i_batch, (X, y) in enumerate(dataloader):
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            outputs = self(X)
            error = self.criterion(outputs, y)
            self.backprop(error)
            running_error += error.detach().cpu().numpy().item() # make it a float

            if i_batch % every_batch == 0:
                first = max(0, i_batch - every_batch)
                print('Error of batches {} to {}:'.format(first, i_batch), running_error/(X.shape[0]*(every_batch)))
                print('Trained {} Batches of size {}!'.format(i_batch, X.shape[0]))
                if save_path:
                        torch.save(self.state_dict(), save_path)
                running_error = 0





    def backprop(self, error, learning_rate=0.01):
        self.zero_grad()
        error.backward(retain_graph=False)

        #update

        # for f in self.parameters():
            # f.data.sub_(f.grad.data * learning_rate)
        self.optim.step()
