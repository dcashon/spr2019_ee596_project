import numpy as np


def batcher(batch_num, batch_size, data_path):
    data = np.load(data_path / ('train_img_pixnum2000_' + str(batch_num) + '.npy'))
    labels = np.load(data_path / ('train_label_pixnum2000_' + str(batch_num) + '.npy'))
    slides = len(data) // batch_size
    start, end = 0, batch_size
    for i in range(slides + 1):
        if i == slides + 1:
            yield data[start:], labels[start:]
        else:
            yield data[start:end], labels[start:end]
        start += batch_size
        end += batch_size
