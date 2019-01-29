#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:53:31 2018

@author: lpupp
"""

#import os
import cv2
import numpy as np
from dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def dict_map(d, f):
    return dict((k, f(v)) for k, v in d.items())


def minibatch_call(data, nn_model, mb=32):
    """Call model with mini-batches."""
    out = []
    for i in range(math.ceil(data.shape[0]/mb)):
        out.append(nn_model(data[i*mb:(i+1)*mb]))
    return torch.cat(out, dim=0)


def find_top_n_similar(source_embeds, db_embeds, n=1):
    out = {}
    for i in range(source_embeds.shape[0]):
        out[i] = find_top_n_similar_by_img(torch.squeeze(source_embeds[i]), db_embeds, n=n)
    return out


def find_top_n_similar_by_img(embed, db_embeds, n=1):
    sim = []
    for i in range(db_embeds.shape[0]):
        sim.append(as_np(F.cosine_similarity(embed, torch.squeeze(db_embeds[i]), dim=0)).item())

        # TODO (lpupp) test this
        #sim.append(F.cosine_similarity(embed, torch.squeeze(db_embeds[i]), dim=0)
        #sim = as_np(torch.cat(sim, dim=1))
    return sorted(list(range(len(sim))), key=lambda i: sim[i])[-n:]


def set_param_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes=None):
    """TODO (lpupp)."""
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        model_ft = nn.Sequential(
            *list(model_ft.features.children())[:-1],
            nn.MaxPool2d(kernel_size=14)
            )
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model_ft, True)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def plot_outputs(img_ix, similar_ix, imgs, src_style='A', path=None):
    similar_ix.reverse()
    if len(imgs) == 3:
        orig, trans, comp = imgs
    else:
        raise ValueError

    #img_orig = as_np(orig[img_ix]).transpose(1, 2, 0)
    img_orig = orig[img_ix].transpose(1, 2, 0)

    #img_tran = as_np(trans[img_ix]).transpose(1, 2, 0)
    #img_tran = cv2.resize(img_tran, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    img_tran = trans[img_ix].transpose(1, 2, 0)

    #imgs_comp = as_np(comp[similar_ix])
    #imgs_comp = [imgs_comp[i].transpose(1, 2, 0) for i in range(imgs_comp.shape[0])]
    imgs_comp = [comp[i].transpose(1, 2, 0) for i in similar_ix]

    img_out = np.hstack((img_orig, img_tran, *imgs_comp))

    if path is not None:
        filename = str(img_ix) + src_style + '.jpg'
        img_save = (img_out * 255.).astype(np.uint8)
        scipy.misc.imsave(os.path.join(path, filename), img_save)

    return img_out


def plot_all_outputs(similar_ixs, imgs, src_style='A', path=None):
    out = []
    for i in similar_ixs:
        out.append(plot_outputs(i, similar_ixs[i], imgs, src_style, path))
    return out
