'''
Using OpenCV takes a mp4 video and produces a number of images.
Requirements
----
You require OpenCV 3.2 to be installed.
Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py
Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

d = {}

for i, filename in enumerate(sorted(os.listdir("videos"))):
    if filename == ".DS_Store":
        print(filename)
        continue
    print(filename)
    if int(filename[4:8]) > 4462:
        minFrames = int(input('minFrames: '))
    elif int(filename[4:8]) <= 4421:
        minFrames = 450
        print('minFrames:', minFrames)
    else:
        minFrames = 600
        print('minFrames:', minFrames)

    cap = cv2.VideoCapture('videos/' + filename)

    if minFrames > cap.get(7):
        print("{} is too short!! {}x{}".format(filename, cap.get(3), cap.get(4)))
        raise ValueError
    else:
        print("{} is fine!! {}x{}".format(filename, cap.get(3), cap.get(4)))

    abc = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ['SCH', 'CH']
    if int(filename[4:8]) > 4462:
        letter = input('Letter: ').upper()
    elif int(filename[4:8]) <= 4421:
        letter = abc[(i-1)%28]
        print('Letter:', letter)
    else:
        letter = abc[int(filename[4:8])-4435]
        print('Letter:', letter)
        
    while True:
        d[letter] = d.get(letter, -1) + 1
        ret, frame = cap.read()
        cv2.imwrite('dataset/' + letter + str(d[letter]) + '.jpg', frame)
        if cap.get(1) == minFrames:
            break
    print('Converted: ' + filename)
    cap.release()

cv2.destroyAllWindows()


# # Playing video from file:
# cap = cv2.VideoCapture('example.mp4')

# try:
#     if not os.path.exists('data'):
#         os.makedirs('data')
# except OSError:
#     print ('Error: Creating directory of data')

# currentFrame = 0
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Saves image of the current frame in jpg file
#     name = './data/frame' + str(currentFrame) + '.jpg'
#     print ('Creating...' + name)
#     cv2.imwrite(name, frame)

#     # To stop duplicate images
#     currentFrame += 1

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()