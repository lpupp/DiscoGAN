#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 18 2018

Removes "edges" part of images in dataset.

@author: lucagaegauf
"""

import sys
import scipy.misc
import numpy as np
from glob import glob


def extract_imgs(dataset):
    for datatype in ['train', 'val']:
        path = glob('./datasets/%s/%s/*' % (dataset, datatype))
    
        for img in path:
            img_ = scipy.misc.imread(img, mode='RGB').astype(np.float)
            h, w, _ = img_.shape
            if h < w:
                img_ = img_[:, int(w/2):, :]
                scipy.misc.imsave(img, img_)


if __name__ == '__main__':
    dataset = sys.argv[1]
    extract_imgs(dataset)

