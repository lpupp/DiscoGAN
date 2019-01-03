import os
import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize
import scipy.io


#dataset_path = './datasets/'
#dataset_path = '/Users/lucagaegauf/Documents/GitHub/Keras-GAN/discogan/datasets/'
dataset_path = 'C:\\Users\\lucag\\OneDrive\\Documents\\GitHub\\Keras-GAN\\discogan\\datasets'

#celebA_path = os.path.join(dataset_path, 'celebA')
#facescrub_path = os.path.join(dataset_path, 'facescrub')
#chair_path = os.path.join(dataset_path, 'rendered_chairs')
#face_3d_path = os.path.join(dataset_path, 'PublicMM1', '05_renderings')
#face_real_path = os.path.join(dataset_path, 'real_face')
#car_path = os.path.join(dataset_path, 'data', 'cars')

handbag_path = os.path.join(dataset_path, 'edges2handbags')
shoe_path = os.path.join(dataset_path, 'edges2shoes')

table_path = os.path.join(dataset_path, 'furniture/tables')
seating_path = os.path.join(dataset_path, 'furniture/seating')

def shuffle_data(da, db):
    a_idx = list(range(len(da)))
    np.random.shuffle(a_idx)

    b_idx = list(range(len(db)))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[np.array(a_idx)]
    shuffled_db = np.array(db)[np.array(b_idx)]

    return shuffled_da, shuffled_db

def read_images(filenames, image_size=64):

    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(image)

    images = np.stack(images)
    return images

def read_attr_file(attr_path, image_dir):
    f = open(attr_path)
    lines = f.readlines()
    lines = map(lambda line: line.strip(), lines)
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame(items, columns=columns)
    df['image_path'] = df['image_path'].map(lambda x: os.path.join(image_dir, x))

    return df


def get_edge2photo_files(item='edges2handbags', test=False):
    if item == 'edges2handbags':
        item_path = handbag_path
    elif item == 'edges2shoes':
        item_path = shoe_path

    if test:
        item_path = os.path.join(item_path, 'val')
    else:
        item_path = os.path.join(item_path, 'train')

    image_paths = list(map(lambda x: os.path.join(item_path, x), os.listdir(item_path)))

    if test:
        return [image_paths, image_paths]
    else:
        n_slice = int(len(image_paths)/2)
        return [image_paths[:n_slice], image_paths[n_slice:]]

def get_furniture_files(item='tables', test=False):
    if item == 'tables':
        item_path = table_path
    elif item == 'seating':
        item_path = seating_path

    if test:
        pass
    else:
        pass

    image_paths = map(lambda x: os.path.join(item_path, x), os.listdir(item_path))

    if test:
        return [image_paths, image_paths]
    else:
        n_images = len(image_paths)
        return [image_paths[:n_images/2], image_paths[n_images/2:]]
