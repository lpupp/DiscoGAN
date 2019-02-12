import os
os.chdir('/Users/lucagaegauf/Documents/Github/DiscoGAN')
import sys
sys.path.append('discogan')
sys.path.append('/Users/lucagaegauf/Dropbox/GAN/')

#import argparse
from itertools import product

import random
import torch
import torch.nn as nn
import torchvision.models as models

from model import *
from utils import *
from data_utils import *

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})

#parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
#add_bool_arg(parser, 'cuda')

#parser.add_argument('--domain', type=str, default='fashion', help='Set data domain. Choose among `fashion` or `furniture`')
#parser.add_argument('--topn', type=int, default=5, help='load iteration suffix')

#parser.add_argument('--model_path', type=str, default='./final_models/', help='Set the path for trained models')
#parser.add_argument('--topn_path', type=str, default='./top5/', help='Set the path the top5 images will be saved')

#parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')
#parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
#parser.add_argument('--embedding_encoder', type=str, default='vgg19', help='choose among pre-trained alexnet/vgg{11, 13, 16, 19}/vgg{11, 13, 16, 19}bn/resnet{18, 34, 50, 101, 152}/squeezenet{1.0, 1.1}/densenet{121, 169, 201, 161}/inceptionv3 models')
#parser.add_argument('--similarity_metric', type=str, default='cosine', help='choose among cosine/euclidean similarity metrics.')

#parser.add_argument('--image_path', type=str, default=None, help='If provided, single_image will be execute, else main. Path to image.')
#parser.add_argument('--image_class', type=str, default=None, help='Class to with image_path image belongs. E.g. shoes')

#parser.add_argument('--eval_task', type=str, default='out', help='Evaluate on `in` or `out`-of-sample data, `random` for random, or `single` for single image eval.')
#parser.add_argument('--seed', type=int, default=0, help='Random seed')


domain = 'fashion'
model_path = './final_models/'
image_size = 64
model_arch = 'discogan'

domain_d = {'furniture': ['seating', 'tables', 'storage'],
            'fashion': ['handbags', 'shoes', 'belts', 'dresses']}
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def create_nms(task_name, domain2label):
    A, B = task_name.split('2')
    return domain2label[B] + domain2label[A], domain2label[A] + domain2label[B]


def eval_full_domain_set_out(cuda, encoder, model_arch, img_size, topn, domain, paths, enc_img_size):
    """Main TODO.

    Steps:
    - Load data.
    - Load generators.
    - Translate all images with generators.
    - Encode images with pretrained encoder.
    - Get top n.
    - Plot everything.

    TODO(lpupp)
    [ ] Load model ix. Recommendation is to put the final models in './final_models/'
    [ ] Do real-world run through with a zalando image. Compare our output with
        their output. This would require a harvesting of the zalando database
        for a (at least) a few products...
    [ ] Compare topn with and without source class
    """

    d_nm = domain
    domain_set = domain_d[d_nm]
    domain_labs = letters[:len(domain_set)]

    domain = dict((k, v) for k, v in zip(domain_labs, domain_set))
    domain2lab = dict((v, k) for k, v in domain.items())

    label_perms = [i + j for i, j in product(domain_labs, domain_labs) if i != j]

    img_paths_A = get_photo_files(domain['A'])[1]
    img_paths_B = get_photo_files(domain['B'])[1]
    imgs_A = read_images(img_paths_A, img_size)
    imgs_B = read_images(img_paths_B, img_size)

    task_name = 'shoes2handbags'

    path = 'models/fashion/shoes2handbags/discogan64'
    ix = max([float(e.split('-')[1]) for e in os.listdir(path) if 'model_gen' in e])
    generator = torch.load(os.path.join(path, 'model_gen_B-' + str(ix)), map_location={'cuda:0': 'cpu'})

    imgs_B = torch_cuda(imgs_B, False)
    imgs_BA = as_np(generator(imgs_B))

    # Flatten images
    A_flat = np.flatten(imgs_A)
    BA_flat = np.flatten(imgs_BA)
    
    A_flat_std = StandardScaler().fit_transform(A_flat)
    BA_flat_std = StandardScaler().fit_transform(BA_flat)

    pca = PCA(n_components=2)
    pc_A = pca.fit_transform(A_flat_std)
    pc_BA = pca.fit_transform(BA_flat_std)
    
    plt.figure()
    plt.plot(pc_A[0], pc_A[1], 'ko', label='A', markersize=0.5)
    plt.plot(pc_BA[0], pc_BA[1], 'ro', label='BA', markersize=0.5)
    plt.ylabel('loss')
    plt.legend()
    #plt.title('')
    #plt.tight_layout()
    plt.savefig()
    plt.close()
    
    

    
    

    


def main(args):

    cuda = args.cuda
    print('cuda: {}'.format(cuda))

    # TODO(lpupp) Do this in general
    if args.embedding_encoder != 'vgg19':
        raise NotImplementedError
    else:
        #embedding_encoder, enc_input_size = initialize_model(args.embedding_encoder)
        print('loading embedding encoder {}'.format(args.embedding_encoder))
        embedding_encoder = models.vgg19(pretrained=True)
        #embedding_encoder = txt2model[args.embedding_encoder]
        set_param_requires_grad(embedding_encoder, feature_extracting=True)
        embedding_encoder = nn.Sequential(
                *list(embedding_encoder.features.children())[:-1],
                nn.MaxPool2d(kernel_size=14)
                )
    enc_input_size = 224

    if cuda:
        embedding_encoder = embedding_encoder.cuda()

    embedding_encoder.eval()

    model_arch = args.model_arch + str(args.image_size)

    paths = {'model': os.path.join(args.model_path, args.domain),
             'topn': os.path.join(args.topn_path, args.domain),
             'image': args.image_path}

    if args.eval_task == 'single':
        if args.image_path:
            if args.image_class not in domain_d[args.domain]:
                raise ValueError
            eval_single_image(img_class=args.image_class,
                              img_size=args.image_size,
                              topn=args.topn,
                              encoder=embedding_encoder,
                              cuda=cuda,
                              model_arch=model_arch,
                              domain=args.domain,
                              paths=paths,
                              enc_img_size=enc_input_size)
        else:
            raise ValueError
    elif args.eval_task == 'out':
        eval_full_domain_set_out(cuda=cuda,
                                 encoder=embedding_encoder,
                                 model_arch=model_arch,
                                 img_size=args.image_size,
                                 topn=args.topn,
                                 domain=args.domain,
                                 paths=paths,
                                 enc_img_size=enc_input_size)
    elif args.eval_task == 'in':
        eval_full_domain_set_in(cuda=cuda,
                                encoder=embedding_encoder,
                                model_arch=model_arch,
                                img_size=args.image_size,
                                topn=args.topn,
                                domain=args.domain,
                                paths=paths,
                                enc_img_size=enc_input_size)
    elif args.eval_task == 'random':
        random.seed(args.seed)
        eval_random(img_size=args.image_size,
                    topn=args.topn,
                    domain=args.domain,
                    paths=paths)
    else:
        raise ValueError


if __name__ == "__main__":

    global args
    args = parser.parse_args()
    main(args)
