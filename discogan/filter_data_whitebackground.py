#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 24 2018

@author: lucagaegauf

Source:
https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
"""

import os
from glob import glob
import scipy.misc
import numpy as np

import scipy
import scipy.misc
import scipy.cluster

#import matplotlib.pyplot as plt

#%%
NN_ROOT = os.path.abspath('./Documents/Github/Keras-GAN/discogan/datasets/furniture')
os.chdir(NN_ROOT)
os.getcwd()

#%%
NUM_CLUSTERS = 5
n = 50

#%%
datasets = ['tables', 'seating']
for dataset in datasets:
    path = glob('./%s/*' % (dataset))

    to_keep = []
    for i in range(len(path)):
        img = scipy.misc.imread(path[i], mode='RGB').astype(np.float)
        shape = img.shape
        img = img.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        
        codes, dist = scipy.cluster.vq.kmeans(img, NUM_CLUSTERS)
        
        vecs, dist = scipy.cluster.vq.vq(img, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
        
        index_max = scipy.argmax(counts)                    # find most frequent
        peak = codes[index_max]
        #color = ''.join(chr(int(c)) for c in peak).encode('hex')
        color = [int(c) for c in peak]
        #print('image %s: most frequent is %s (#%s)' % (i, peak, color))
        if color[0] + color[1] + color[2] >= 750:
            to_keep.append(path[i])
            print('keep image {}: most frequent is {}'.format(i, color))

    with open('{}_to_keep.txt'.format(dataset), 'w') as f:
        for item in to_keep:
            f.write("%s\n" % item)            

#%%
#i = 0
#img = scipy.misc.imread(path[i], mode='RGB').astype(np.float)       
#plt.imshow(img/255)
