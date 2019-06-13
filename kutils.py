import numpy as np


def batcher(batch_num, batch_size, data_path, classify=True):
    data = np.load(data_path / ('test_img_pixnum2000_' + str(batch_num) + '.npy'))
    if classify is True:
        labels = np.load(data_path / ('test_label_pixnum2000_' + str(batch_num) + '.npy'))
    else:
        # we load the bounding box data instead
        labels = np.load(data_path / ('test_bbox_pixnum2000_' + str(batch_num) + '.npy'))
    slides = len(data) // batch_size
    start, end = 0, batch_size
    for i in range(slides + 1):
        if i == slides + 1:
            yield data[start:], labels[start:]
        else:
            yield data[start:end], labels[start:end]
        start += batch_size
        end += batch_size
