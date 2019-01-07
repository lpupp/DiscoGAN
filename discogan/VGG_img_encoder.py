#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:53:31 2018

@author: lucagaegauf
"""

#import os
from glob import glob
import scipy.misc
import numpy as np

from keras.applications import vgg19
import sklearn.metrics
import scipy.spatial
import skimage.measure

import matplotlib.pyplot as plt

#%%
n = 50

#%%
data_type = 'train'
dataset = 'edges2handbags'

path = glob('./datasets/%s/%s/*' % (dataset, data_type))

#%%
def cos_sim(x, y):
    return sklearn.metrics.pairwise.cosine_similarity(x.reshape(1, -1),
                                                      y.reshape(1, -1)).item()

def euclid(x, y):
    return scipy.spatial.distance.euclidean(x, y)

#%%
vgg = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='max')

vgg.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#%% Load n images
imgs = []
for i in range(n):
    img = scipy.misc.imread(path[i], mode='RGB').astype(np.float)
    img = img/127.5 - 1
    imgs.append(img)

imgs = np.stack(imgs)

# Generate image embeddings
img_embed = vgg.predict(imgs)
print(img_embed.shape)

#%% Find similarities and most similar
cossim = {}
amax = {}
for target in range(n):
    cossim.setdefault(target, [])
    for i in range(n):
        cossim[target].append(cos_sim(img_embed[target], img_embed[i]))

    top2 = np.array(cossim[target]).argsort()[-2:][::-1]
    top1 = top2[top2 != target]
    amax[target] = top1.item()

#%% Display highest similarity matches for image i
i = 24
k, v = list(amax.keys()), list(amax.values())
plt.imshow(np.concatenate((imgs[k[i]], imgs[v[i]]), axis=1))

#%% ZALANDO DATASET ----------------------------------------------------------
# https://convertio.co/jp2-jpg/

# Load pretrained model with zalando image dims
vggz = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(1100, 762, 3), pooling='max')
# Load zalando images
pathz = glob('./datasets/zalando_test/*')

imgsz = []
for i in pathz:
    img = scipy.misc.imread(i, mode='RGB').astype(np.float)
    img = img/127.5 - 1
    imgsz.append(img)

imgsz = np.stack(imgsz)

# Generate embeddings
img_embedz = vggz.predict(imgsz)
print(img_embedz.shape)

#%%
# Works relatively well: 1, 12, 18, 21
# Possibly doesn't work: 26
target = 21
plt.imshow(imgs[target])
cs = []
for i in range(9):
    cs.append(cos_sim(img_embed[target], img_embedz[i]))

#%%
plt.imshow(imgsz[np.argmax(np.array(cs))])


#%% DISCOGAN STYLE TRANSFER --------------------------------------------------
# I dont know which of these I actually need
from keras.models import load_model
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam

#%%
# Load models
g_AB = load_model('./models/g_AB_0.h5')
g_BA = load_model('./models/g_BA_0.h5')

#%%
i = 1
# max pool image to correct dimension
img_ = skimage.measure.block_reduce(imgs[i], (2, 2, 1), np.max)
img_trans = g_BA.predict(np.expand_dims(img_, axis=0))
plt.imshow(np.concatenate((img_, img_trans[0]), axis=1))


#%% PLOT RESULTS --------------------------------------------------------------

#import os
from glob import glob
import scipy.misc
import numpy as np

import matplotlib.pyplot as plt

n = 21

data_type = 'train'
dataset = 'edges2handbags'

path = glob('../Keras-GAN/datasets/%s/%s/*' % (dataset, data_type))

#%% Load n images
imgs = []
for i in range(n):
    img = scipy.misc.imread(path[i], mode='RGB').astype(np.float)
    img = img/127.5 - 1
    imgs.append(img)

imgs = np.stack(imgs)
input_imgs = imgs[:3]
trans_imgs = imgs[3:6]
trans_imgs = imgs[6:]

#%% Display highest similarity matches for image i

plt.imshow(np.concatenate((imgs[k[i]], imgs[v[i]]), axis=1))
