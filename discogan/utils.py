#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:53:31 2018

@author: lpupp
"""

import cv2
import numpy as np
from dataset import *

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image, ImageOps, ImageDraw  # ImageFont

from data_utils import as_np


def torch_cuda(x, cuda):
    """Convert input to (cuda) tensor."""
    with torch.no_grad():
        x = Variable(torch.FloatTensor(x))
    if cuda:
        x = x.cuda()
    return x


def dict_map(d, f):
    """Apply function on every value in dictionary."""
    return dict((k, f(v)) for k, v in d.items())


def minibatch_call(dt, nn_model, mb=32):
    """Call model with mini-batches."""
    out = []
    dim = dt.shape[0]
    for i in range(math.ceil(dim/mb)):
        try:
            out.append(nn_model(dt.narrow(0, i*mb, mb)))
        except:
            out.append(nn_model(dt.narrow(0, i*mb, dim - i*mb)))
    return torch.cat(out, dim=0)


def set_param_requires_grad(model, feature_extracting):
    """Set which parameters require a gradient for training."""
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


def _find_top_n_similar(source_embeds, db_embeds, n=1):
    out = {}
    for i in range(source_embeds.shape[0]):
        out[i] = _find_top_n_similar_by_img(torch.squeeze(source_embeds[i]), db_embeds, n=n)
    return out


def _find_top_n_similar_by_img(embed, db_embeds, n=1):
    sim = []
    for i in range(db_embeds.shape[0]):
        sim.append(as_np(F.cosine_similarity(embed, torch.squeeze(db_embeds[i]), dim=0)).item())

        # TODO (lpupp) test this
        #sim.append(F.cosine_similarity(embed, torch.squeeze(db_embeds[i]), dim=0)
        #sim = as_np(torch.cat(sim, dim=1))
    return sorted(list(range(len(sim))), key=lambda i: sim[i])[-n:]


def find_top_n_similar(src_embeds, db_embeds, n=1):
    """Find top n similar for each image in array."""
    out = {}
    for i in range(src_embeds.shape[0]):
        out[i] = find_top_n_similar_by_img(torch.squeeze(src_embeds[i]), db_embeds, n=n)
    return out


def find_top_n_similar_by_img(embed, db_embeds, n=1):
    """Find top n similar images."""
    sim = {}
    for i in range(db_embeds.shape[0]):
        sim[i] = as_np(F.cosine_similarity(embed, torch.squeeze(db_embeds[i]), dim=0)).item()

    return sorted(sim.items(), key=lambda kv: kv[1])[-n:]


def plot_overall(similar_ix, img_src, img_trans, img_db, img_ix=0, path=None):
    """Plot top n recommendations across all comparison categories."""
    similar_ix.reverse()
    dom = [e[0] for e in similar_ix]
    ixs = [e[1][0] for e in similar_ix]
    scores = [e[1][1] for e in similar_ix]
    n_class = len(img_trans) + 1

    img_orig = img_src[0].transpose(1, 2, 0)
    imgs_tran = [v[0].transpose(1, 2, 0) for v in img_trans.values()]
    imgs_comp = [img_db[d][i].transpose(1, 2, 0) for d, i in zip(dom, ixs)]
    img_out = np.hstack((img_orig, *imgs_tran, *imgs_comp))

    if path:
        col = (0, 0, 0)
        n = len(scores)
        filename = str(img_ix) + 'all.jpg'

        img_save = Image.fromarray((img_out * 255.).astype(np.uint8))
        img_save = ImageOps.expand(img_save, border=20, fill='white')
        draw = ImageDraw.Draw(img_save)
        draw.text((20, 2), 'orig', col)
        draw.text((20 + 64, 2), 'trans', col)
        draw.text((20 + 64*(n_class), 2), 'top {} recommendations'.format(n), col)
        for i, sim in enumerate(similar_ix):
            draw.text((64*(i+n_class)+20, 88), '({} {})'.format(ixs[i], round(scores[i], 3)), col)
        img_save.save(os.path.join(path, filename))

    return img_out


def plot_outputs(img_ix, similar_ix, imgs, src_style='A', path=None):
    """Plot top n recommendations for each categories individually."""
    similar_ix.reverse()
    ixs = [e[0] for e in similar_ix]
    scores = [e[1] for e in similar_ix]

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
    imgs_comp = [comp[i].transpose(1, 2, 0) for i in ixs]

    img_out = np.hstack((img_orig, img_tran, *imgs_comp))

    if path:
        col = (0, 0, 0)
        n = len(scores)
        filename = str(img_ix) + src_style + '.jpg'
        #img_save = (img_out * 255.).astype(np.uint8)
        #scipy.misc.imsave(os.path.join(path, filename), img_save)

        img_save = Image.fromarray((img_out * 255.).astype(np.uint8))
        img_save = ImageOps.expand(img_save, border=20, fill='white')
        draw = ImageDraw.Draw(img_save)
        draw.text((20, 2), 'orig', col)
        draw.text((20 + 64, 2), 'trans', col)
        draw.text((20 + 64*2, 2), 'top {} recommendations'.format(n), col)
        for i, sim in enumerate(similar_ix):
            draw.text((64*(i+2)+20, 88), '({} {})'.format(sim[0], round(sim[1], 3)), col)
        img_save.save(os.path.join(path, filename))

    return img_out


def plot_all_outputs(similar_ixs, imgs, src_style='A', path=None):
    """plot_outputs for array of source images."""
    out = []
    for i in similar_ixs:
        out.append(plot_outputs(i, similar_ixs[i], imgs, src_style, path))
    return out
