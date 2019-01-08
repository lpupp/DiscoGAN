#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:53:31 2018

@author: lucagaegauf
"""

#import os
import numpy as np
from dataset import *


def img4save(data):
    data_ = as_np(data).transpose(1, 2, 0) * 255.
    return data_.astype(np.uint8)[:, :, ::-1]


def get_data():
    if args.task_name == 'handbags2shoes' or args.task_name == 'shoes2handbags':
        data_A_1, data_A_2 = get_edge2photo_files(item='edges2handbags', test=False)
        test_A_1, test_A_2 = get_edge2photo_files(item='edges2handbags', test=True)

        data_A = np.hstack([data_A_1, data_A_2])
        test_A = np.hstack([test_A_1, test_A_2])

        data_B_1, data_B_2 = get_edge2photo_files(item='edges2shoes', test=False)
        test_B_1, test_B_2 = get_edge2photo_files(item='edges2shoes', test=True)

        data_B = np.hstack([data_B_1, data_B_2])
        test_B = np.hstack([test_B_1, test_B_2])

    elif args.task_name == 'tables2chairs' or args.task_name == 'chairs2tables':
        data_A_1, data_A_2 = get_furniture_files(item='tables', test=False)
        test_A_1, test_A_2 = get_furniture_files(item='tables', test=True)

        data_A = np.hstack([data_A_1, data_A_2])
        test_A = np.hstack([test_A_1, test_A_2])

        data_B_1, data_B_2 = get_furniture_files(item='seating', test=False)
        test_B_1, test_B_2 = get_furniture_files(item='seating', test=True)

        data_B = np.hstack([data_B_1, data_B_2])
        test_B = np.hstack([test_B_1, test_B_2])

    return data_A, data_B, test_A, test_B
