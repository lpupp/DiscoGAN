#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 18 2018

@author: lpupp

Utils to process the image datasets for training.

Source:
https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
"""

import os
import sys
import cv2
import random
import numpy as np
from glob import glob

import scipy
import scipy.misc
import scipy.cluster


def extract_imgs_from_edges(dataset):
    """Extract images from edges2image files."""
    for datatype in ['train', 'val']:
        path = glob('./datasets/%s/%s/*' % (dataset, datatype))

        for img in path:
            img_ = scipy.misc.imread(img, mode='RGB').astype(np.float)
            h, w, _ = img_.shape
            if h < w:
                img_ = img_[:, int(w/2):, :]
                scipy.misc.imsave(img, img_)


def resize_and_pad(dataset, img_size=256):
    """Pad non-square images and resize to img_size x img_size."""
    path = glob('%s/*' % (dataset))
    for img_pth in path:
        img = cv2.imread(img_pth)
        old_size = img.shape[:2]

        ratio = float(img_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format

        img = cv2.resize(img, (new_size[1], new_size[0]))

        dw = img_size - new_size[1]
        dh = img_size - new_size[0]
        top, bottom = dh//2, dh-(dh//2)
        left, right = dw//2, dw-(dw//2)

        color = [255, 255, 255]
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=color)

        cv2.imwrite(img_pth, new_img)


def filter_white_background(dataset, n=50, NUM_CLUSTERS=5):
    """Filter dataset for those images which are predominately white."""
    path = glob('./%s/*' % (dataset))

    to_keep = []
    for i in range(len(path)):
        img = scipy.misc.imread(path[i], mode='RGB').astype(np.float)
        shape = img.shape
        img = img.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

        codes, dist = scipy.cluster.vq.kmeans(img, NUM_CLUSTERS)

        vecs, dist = scipy.cluster.vq.vq(img, codes)      # Assign codes
        counts, bins = scipy.histogram(vecs, len(codes))  # Count occurrences

        index_max = scipy.argmax(counts)                  # Find most frequent
        peak = codes[index_max]
        color = [int(c) for c in peak]

        if color[0] + color[1] + color[2] >= 750:
            to_keep.append(path[i])
            print('keep image {}: most frequent is {}'.format(i, color))

    with open(dataset + '_to_keep.txt', 'w') as f:
        for item in to_keep:
            f.write('%s\n' % item)


def train_val_sample(dataset, n_val=400, seed=0):
    """Split data into training and validation set."""
    path_train = os.path.join(dataset, 'train')
    path_val = os.path.join(dataset, 'val')
    if not os.path.isdir(path_train):
        os.mkdir(path_train)
    if not os.path.isdir(path_val):
        os.mkdir(path_val)

    paths = glob('%s/*.jpg' % (dataset))

    random.seed(seed)
    if len(paths) != 0:
        random.shuffle(paths)
        train_paths = paths[:-n_val]
        val_paths = paths[-n_val:]
        for train_path in train_paths:
            path, file = os.path.split(train_path)
            os.rename(train_path, os.path.join(path, 'train', file))
        for val_path in val_paths:
            path, file = os.path.split(val_path)
            os.rename(val_path, os.path.join(path, 'val', file))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError
    if len(sys.argv) >= 3:
        dataset, type = sys.argv[1], sys.argv[2]
        if len(sys.argv) == 4:
            dim = sys.argv[3]
        else:
            dim = None

    if type == 'extract':
        extract_imgs_from_edges(dataset)
    elif type == 'resize' or type == 'pad':
        resize_and_pad(dataset, img_size=dim or 256)
    elif type == 'filter':
        filter_white_background(dataset)
    elif type == 'sample':
        train_val_sample(dataset, n_val=dim or 1000)
    else:
        raise NotImplementedError
