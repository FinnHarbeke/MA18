import torch
from net import FingeralphabetNet
from preprocess import FingeralphabetDataset
import os
from torch.utils.data import DataLoader
import torchvision as tv
import pandas as pd

nn_subfolder = '8.try'
test_subfolder = '8.try'

def confusion_matrix(nn, img_dir):
    nn.train(False)
    dataset = tv.datasets.ImageFolder('../dataset/test2', transform=tv.transforms.Compose([
    tv.transforms.Resize((160, 120)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([.5]*3, [.5]*3),
]))
    dataloader = DataLoader(dataset, 
                    batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
    confusion_matrix = pd.DataFrame(0, columns=abc, index=abc)
    acc = 0
    for i_batch, (X, y) in enumerate(dataloader):
        outputs = nn(X)
        target_inds = y
        for i in range(outputs.shape[0]):
            output_ls = [float(x) for x in outputs[i]]
            output_ind = output_ls.index(max(output_ls))
            target_ind = int(y[i])
            output_letter = abc[output_ind]
            target_letter = abc[target_ind]
            confusion_matrix.at[target_letter, output_letter] += 1
            if target_ind == output_ind:
                acc += 1
            
    count = len(dataset)
    return confusion_matrix, acc/count

for i in range(21):
    fn = 'Nets/' + nn_subfolder + '/' + str(i) + '.pth'
    nn = FingeralphabetNet()
    if torch.cuda.is_available():
        nn = nn.cuda()
    nn.load_state_dict(torch.load(fn, map_location=lambda storage, loc: storage))
    data, acc = confusion_matrix(nn, '../dataset/test2')
    data.to_csv('Nets/testResults/' + test_subfolder + '/ep{}.csv'.format(i))
    print('Created Nets/testResults/' + test_subfolder + '/ep{}.csv!'.format(i))
    print('Acc: {}'.format(acc))
print("FINISHED")
