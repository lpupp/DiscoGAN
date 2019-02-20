#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 12 2019

@author: lpupp
"""

import os
import argparse
# import cv2
import numpy as np

from PIL import Image, ImageOps, ImageDraw

from glob import glob

# TODO:
# [ ] test trim_border
# [ ] test drop_src_image
# [ ] where do we need to pass in args?


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')

parser.add_argument('--model_arch', type=str, default='discogan', help='Model architecture.')
parser.add_argument('--topn_path', type=str, default='./top5/', help='Set the path the result images will be saved.')
parser.add_argument('--domain', type=str, default='fashion', help='Set domain name')
parser.add_argument('--task', type=str, default=None, help='Set domain categories.')
parser.add_argument('--image_size', type=int, default=64, help='Original input image dimension.')


def trim_border(img):
    """Trim border."""
    # TODO(lpupp) Crop is an ugly function...
    w, h = img.size
    if h > args.image_size:
        img = img.crop((0, 20, w, h-20))
    if w % args.image_size != 0:
        img = img.crop((20, 0, w-20, h))
    return img


def drop_src_image(img):
    """Remove original image."""
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20, 0), (20+args.image_size*2, 20+args.image_size)],
                   fill=(255, 255, 255))
    return img


def remove_scores(img):
    """Remove original image."""
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20+args.image_size*2, 20+args.image_size),
                    (40+args.image_size*7, 40+args.image_size)],
                   fill=(255, 255, 255))
    return img


domain_d = {'furniture': ['seating', 'tables', 'storage'],
            'fashion': ['handbags', 'shoes', 'belts', 'dresses']}


def main(args):
    task = args.task
    domain = args.domain
    if task:
        tasks = [task]
        if task not in domain_d[domain]:
            raise ValueError
    else:
        print('`task` was not provided. Summarizing image in all ' + domain + ' categories.')
        tasks = domain_d[domain]

    for tsk in tasks:
        cdir = os.path.join(args.topn_path, domain, tsk)
        if not os.path.exists(cdir):
            print(tsk + ' does not exist. Proceeding to next category.')
            continue

        sum_dir = os.path.join(cdir, 'summary')
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)

        # All dirs in cdir if in eval_dirs
        tsk_dirs = [e for e in os.listdir(cdir) if e in eval_dirs]

        # All images in all tsk_dirs
        tsk_img_paths = [glob(os.path.join(cdir, e, '*')) for e in tsk_dirs]
        tsk_img_paths = [j for i in tsk_img_paths for j in i]

        # Unique image names from tsk_img_paths
        img_nms = list(set([os.path.split(e)[1] for e in tsk_img_paths]))
        # Image paths grouped by image name
        grouped_imgs = dict((nm, [e for e in tsk_img_paths if '\\'+nm in e]) for nm in img_nms)

        for k, v in grouped_imgs.items():
            print(k)
            imgs = []
            for i, pth in enumerate(v):
                # otherwise: img = cv2.imread(pth)
                img = Image.open(pth)
                if img is not None:
                    #img = trim_border(img)
                    img = remove_scores(img)
                    if i != 0:
                        img = drop_src_image(img)

                    #img = ImageOps.expand(img, border=2, fill=(255, 255, 255))
                    imgs.append(img)

            if len(imgs) > 0:
                imgs = np.vstack(imgs)
                # otherwise: imgs = np.vstack((*imgs))

                # Might not be needed...
                img_save = Image.fromarray(imgs.astype(np.uint8))
                img_save.save(os.path.join(sum_dir, k))
                # otherwise: img_save.save(os.path.join(sum_dir, k + '.jpg'))


if __name__ == "__main__":

    global args
    args = parser.parse_args()

    eval_dirs = [args.model_arch + str(args.image_size) + '_out',
                 args.model_arch + str(args.image_size) + '_in',
                 args.model_arch + str(args.image_size) + '_out_narrow',
                 args.model_arch + str(args.image_size) + '_in_narrow',
                 'vgg_out',
                 'vgg_in',
                 'random']

    main(args)
