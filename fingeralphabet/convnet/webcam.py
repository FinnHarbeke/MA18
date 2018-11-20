from net import FingeralphabetNet
import torch
import torchvision as tv
import cv2
from PIL import Image

def letter(output, count):
    """
    returns the top count letters of an output of the net
    """
    output = [float(x) for x in output[0]]
    maxes = [None for i in range(count)] 
    indexes = [None for i in range(count)]
    for i, x in enumerate(output):
        if None in maxes:
            indexes[maxes.index(None)] = i
            maxes[maxes.index(None)] = x
        elif x > min(maxes):
            indexes[maxes.index(min(maxes))] = i
            maxes[maxes.index(min(maxes))] = x
    abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["SCH", "CH", "NOTHING"]
    # sort
    res_maxes = []
    res_indexes = []
    for i in range(count):
        res_maxes.append(max(maxes))
        res_indexes.append(indexes[maxes.index(max(maxes))])
        indexes.remove(indexes[maxes.index(max(maxes))])
        maxes.remove(max(maxes))
    return [abc[i] for i in res_indexes], res_maxes

def webcamNN(nn):
    nn.train(False)
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture()
    vc.open(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        # resize
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        transforms = tv.transforms.Compose([
            tv.transforms.Resize((160, 120)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([.5]*3, [.5]*3),
        ])
        img = transforms(img).float()

        output = nn(img.unsqueeze(0))
        lets, scores = letter(output, 3)
        res = []
        for i in range(len(lets)):
            res.append("{}.: {} ({})".format(i+1, lets[i], round(scores[i], 2)))
        print("".join(res))

        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")

if __name__ == "__main__":
    nn = FingeralphabetNet()
    nn.load_state_dict(torch.load("Nets/8.try/8.pth", map_location=lambda storage, loc: storage))
    webcamNN(nn)
