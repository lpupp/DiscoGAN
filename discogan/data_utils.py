#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 18 2018

@author: lpupp
"""

import os
import numpy as np
#from dataset import *
from glob import glob

#import csv
import cv2
#import pandas as pd
#from scipy.misc import imresize
#import scipy.io

dataset_path = '/Users/lucagaegauf/Dropbox/GAN'
#dataset_path = 'C:\\Users\\lucag\\Dropbox\\GAN'

fashion_path = os.path.join(dataset_path, 'fashion')
furniture_path = os.path.join(dataset_path, 'furniture')

paths = {'handbags': os.path.join(fashion_path, 'handbags'),
         'shoes': os.path.join(fashion_path, 'shoes'),
         'belts': os.path.join(fashion_path, 'belts'),
         'dresses': os.path.join(fashion_path, 'dresses'),
         'tables': os.path.join(furniture_path, 'tables'),
         'seating': os.path.join(furniture_path, 'seating'),
         'storage': os.path.join(furniture_path, 'storage'),
         'carpets': os.path.join(furniture_path, 'carpets')}


def shuffle_data(da, db):
    """Shuffle datasets A and B."""
    a_idx = list(range(len(da)))
    np.random.shuffle(a_idx)

    b_idx = list(range(len(db)))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[np.array(a_idx)]
    shuffled_db = np.array(db)[np.array(b_idx)]

    return shuffled_da, shuffled_db


def read_image(filename, image_size=64):
    """Read image from path."""
    image = cv2.imread(filename)
    if image is not None:
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
    return image


def read_images(filenames, image_size=64):
    """Read images from list of paths."""
    images = []
    for fn in filenames:
        image = read_image(fn)
        if image is None:
            continue
        images.append(image)

    images = np.stack(images)
    return images


def as_np(data):
    """Get tensor as numpy."""
    return data.cpu().data.numpy()


def img4save(data):
    """Format images from tensor for saving."""
    data_ = as_np(data).transpose(1, 2, 0) * 255.
    return data_.astype(np.uint8)[:, :, ::-1]


def resize_img(x, dsize):
    """Resize image."""
    return cv2.resize(x.transpose(1, 2, 0), dsize=dsize, interpolation=cv2.INTER_CUBIC)


def resize_array_of_images(v, dsize):
    """Resize array of images."""
    v = np.stack([resize_img(v[i], dsize) for i in range(v.shape[0])])
    return v.transpose(0, 3, 1, 2)


def get_photo_files(nm):
    """Get photo files' paths."""
    train_paths = glob(os.path.join(paths[nm], 'train', '*'))
    val_paths = glob(os.path.join(paths[nm], 'val', '*'))

    return train_paths, val_paths


def get_data(args):
    """Get domain A and B datasets."""
    A, B = args.task_name.split('2')
    data_A, test_A = get_photo_files(A)
    data_B, test_B = get_photo_files(B)
    return data_A, data_B, test_A, test_B
