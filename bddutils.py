import json
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer



class DataHandler():
    """
    Handler for BDD100k Data. The data can be sourced from:

    https://bdd-data.berkeley.edu

    User should configure data paths. See __init__ for assumed locations

    
    Author: D. Cashon
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.train_img = os.listdir(self.data_dir / "images" / "100k"/ "train")
        self.train_img_dir = self.data_dir / "images" / "100k" / "train"
        self.train_labels = json.load(open(self.data_dir / 
        "labels/bdd100k_labels_images_train.json", 'rb'))
        self.classes = ['bike', 'bus', 'car', 'motor', 'person', 'rider',
        'traffic light', 'traffic sign', 'train', 'truck']
    
    




    def get_random(self, num_samples):
        """
        Samples random photos and annotates them from BDD dataset
        """
        plt.clf()
        for idx in random.sample(range(len(self.train_labels)), num_samples):
            temp_img = plt.imread(self.data_dir /"images"/ "100k" / "train" /
                    self.train_labels[idx]['name'])
            fig, ax = plt.subplots(1, figsize=(12,10))
            for objects in self.train_labels[idx]['labels']:
                try:
                    x1 = objects['box2d']['x1']
                    x2 = objects['box2d']['x2']
                    y1 = objects['box2d']['y1']
                    y2 = objects['box2d']['y2']
                except:
                    continue
                ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                     linewidth=1, edgecolor='r', facecolor='None'))
                ax.annotate(objects['category'], color='w', xy=(x1,
                             y1), fontsize=8)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(temp_img)
    
    def bb_to_train(self, size, batch_size, save_dir='./'):
        """
        Parses all training images, looks at all classes present in the image,
        and saves them as individual images for NN training. Computes centroid
        of bounding box and crops the image with the smallest square that
        contains said bounding box. 

        Inputs:
            size:      size for image reshape. Assumed square
            batch_size:use to split up the saved data files due to memory
            requirements

        Outputs:
            None: 

        """
        enc = LabelBinarizer() # to one hot encode
        # note batch_size refers to number of images to process,
        # not necssarily equal ot number of output images 
        num_batches = len(self.train_labels) // batch_size
        # loop through every image
        start = 0
        end = start + batch_size
        for batch_num in range(batch_size): # remainder is ignored
            images, lab, bbox = [], [], []
            error_count = 0 # issue if bbox outside image
            print('Saving Img from: \t ' + str(start) + " to " + str(end))
            for i in range(start, end):
                temp_img = plt.imread(self.train_img_dir /
                        self.train_labels[i]['name']) # load the image
                for item in self.train_labels[i]['labels']:
                    # check to make sure its a relevant cat
                    if (item['category'] in self.classes and not
                    item['attributes']['occluded'] and not
                    item['attributes']['truncated']):
                        # compute bbox centroid
                        x1 = int(item['box2d']['x1'])
                        y1 = int(item['box2d']['y1'])
                        x2 = int(item['box2d']['x2'])
                        y2 = int(item['box2d']['y2'])
                        x_center = x1 + (x2 - x1) // 2
                        y_center = y1 + (y2 - y1) // 2
                        dim = max((x2 - x1) // 2, (y2 - y1) // 2)
                        top_left_x = x_center - dim
                        top_left_y = y_center - dim
                        # get the image based on centroid box
                        to_reshape = temp_img[(y_center-dim):(y_center+dim),
                            (x_center - dim):(x_center + dim), :]
                        # get the new bounding box w.r.t patch
                        h_r, w_r, c_r = to_reshape.shape
                        try:
                            x1_shift = (x1 - top_left_x) * (size//h_r)
                            y1_shift = (y1 - top_left_y) * (size//w_r)
                            x2_shift = x1_shift + (x2-x1) * (size//h_r)
                            y2_shift = y1_shift + (y2-y1) * (size//w_r)
                            dx = x2_shift - x1_shift
                            dy = y2_shift - y1_shift
                            images.append(cv2.resize(to_reshape, (size, size)))
                            bbox.append(np.array([x1_shift, y1_shift, dx, dy]))
                            # get object integer class label
                            lab.append(self.classes.index(item['category']))
                        except:
                            error_count += 1
            # save
            print('Saving... \n')
            print('Total Images in batch:' + str(len(images)))
            print('Total Num Errors: \t' + str(error_count))
            np.save(open(save_dir + 'train_img_' + str(batch_num) + '.npy', 'wb'), np.array(images))
            np.save(open(save_dir + 'train_label_' + str(batch_num) + '.npy', 'wb'),
                    np.array(lab))
            np.save(open(save_dir + 'train_bbox_' + str(batch_num) + '.npy', 'wb'),
                    np.array(bbox))
            start += batch_size
            end += batch_size


