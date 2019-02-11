#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 11 2019

@author: lpupp
"""

import unittest

from utils import *

import numpy as np


# https://www.youtube.com/watch?v=1Lfv5tUGsn8&vl=en
# to run:
# cd Documents/Github/DiscoGAN/discogan
# python -m unittest utils_test.py
'''
###############################################################################
#                         FUNCTIONS THAT NEED TESTING                         #
###############################################################################

def find_top_n_similar(src_embeds, db_embeds, n=1):
    """Find top n similar for each image in array."""
    out = {}
    for i in range(src_embeds.shape[0]):
        out[i] = find_top_n_similar_by_img(torch.squeeze(src_embeds[i]), db_embeds, n=n)
    return out


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

    img_orig = orig[img_ix].transpose(1, 2, 0)
    img_tran = trans[img_ix].transpose(1, 2, 0)
    imgs_comp = [comp[i].transpose(1, 2, 0) for i in ixs]

    img_out = np.hstack((img_orig, img_tran, *imgs_comp))

    if path:
        col = (0, 0, 0)
        n = len(scores)
        filename = str(img_ix) + src_style + '.jpg'

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

###############################################################################
#                                     END                                     #
###############################################################################
'''

class UtilsTest(unittest.TestCase):
    """Test module for utils.py"""

    def test_find_top_n_similar_by_img(self):
        """Test whether..."""
        vec = np.array([1., 1., 1.])
        vec = torch_cuda(vec, False)
        comp_vec = np.array([[1., 1., 0.9], [0.9, 1., 2.], [-1., 5., 0.1]])
        comp_vec = torch_cuda(comp_vec, False)

        top1 = find_top_n_similar_by_img(vec, comp_vec, n=1)
        top2 = find_top_n_similar_by_img(vec, comp_vec, n=2)
        top3 = find_top_n_similar_by_img(vec, comp_vec, n=3)
        top4 = find_top_n_similar_by_img(vec, comp_vec, n=4)

        self.assertListEqual([e[0] for e in top1], [0])
        self.assertListEqual([e[0] for e in top2], [1, 0])
        self.assertListEqual([e[0] for e in top3], [2, 1, 0])
        self.assertListEqual([e[0] for e in top4], [2, 1, 0])

        #self.assertDictEqual()
        #self.assertAlmostEqual()
        #self.assertListEqual()
        #self.assertEqual()


if __name__ == '__main__':
    unittest.main()
