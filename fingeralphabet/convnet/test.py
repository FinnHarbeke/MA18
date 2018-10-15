import torch
from net import FingeralphabetNet
from preprocess import FingeralphabetDataset
import os
from torch.utils.data import DataLoader
import pandas as pd

nn_subfolder = '3.try'
test_subfolder = '3.try'

def confusion_matrix(nn, img_dir):
    nn.train(False)
    dataset = FingeralphabetDataset(img_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
    confusion_matrix = pd.DataFrame(0, columns=abc, index=abc)
    acc = 0
    for i_batch, sample_batched in enumerate(dataloader):
        outputs = nn(sample_batched['image_tensor'])
        target_inds = sample_batched['target_ind']
        for i in range(outputs.shape[0]):
            output_ls = list(outputs[i])
            output_ind = output_ls.index(max(output_ls))
            target_ind = int(target_inds[i])
            output_letter = abc[output_ind]
            target_letter = abc[target_ind]
            confusion_matrix.at[target_letter, output_letter] += 1
            if target_ind == output_ind:
                acc += 1
            
    count = len(os.listdir(img_dir))
    return confusion_matrix, acc/count

for i in range(31):
    fn = 'Nets/' + nn_subfolder + '/' + str(i) + '.pth'
    nn = FingeralphabetNet()
    if torch.cuda.is_available():
        nn = nn.to(device=torch.device('cuda:0'))
    nn.load_state_dict(torch.load(fn, map_location=lambda storage, loc: storage))
    data, acc = confusion_matrix(nn, '../dataset/test/test')
    data.to_csv('Nets/testResults/' + test_subfolder + '/ep{}.csv'.format(i))
    print('Created Nets/testResults/' + test_subfolder + '/ep{}.csv!'.format(i))
    print('Acc: {}'.format(acc))
print("FINISHED")
