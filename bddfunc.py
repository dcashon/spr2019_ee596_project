import cv2
import numpy as np



# D. Cashon
# 2019 06 01
# collection of functions for image processing




def sliding_window(img, dx, dy, stride):
    """
    Slides a window of size (dx, dy) across the input image
    (dx, dy) should typically represent the expected input to the network
    Stride indicates the amount of pixels to move at each window jump, this
    impacts runtime

    Inputs:
        -img:       ndarray of image data
        -dx:        width of box
        -dy:        height of box
        -stride:    stride length, assumed to be equal in both dim

    Output:
        -img_slide: ndarray of size [batch_size, window_num, dx, dy, 3] (three
        channel input assumed)
    Is this even necessary? convnet sliding window. explore later
    """

    batch_size, h, w, c = img.shape
    x, y = 0, 0 # starting top left of window
    coordx, coordy, output = [], [], []
    for image in range(batch_size):
        windows = []
        for i in range(0, h-dx+1, stride):
            for j in range(0, w-dy+1, stride):
                coordx.append(i)
                coordy.append(j)
                windows.append(img[image,i:i+dx, j:j+dy, :])
    output.append(np.array(windows))

    return output, coordx, coordy

def training_batcher(batch_num, minibatch_size, mode='classification'):
    if mode is 'classification':
        x1 = np.load('train_img_' + str(batch_num) + '.npy')
        x2 = np.load('train_label_' + str(batch_num) + '.npy')
        start = 0
        end = start + minibatch_size
        todo = len(x1) // minibatch_size + 1 # remainder
        for i in range(todo):
            if i == todo - 1:
                # we are on last batch, use remainder
                yield x1[start:len(x1)], x2[start:len(x1)]
            else:
                yield x1[start:end], x2[start:end]
                start += minibatch_size
                end += minibatch_size
        return 0

