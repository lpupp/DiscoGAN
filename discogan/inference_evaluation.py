import os
import argparse
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

from dataset import *
from model import *
from utils import *
from data_utils import *

import scipy

#import sklearn.metrics.pairwise.cosine_similarity as cosine_similarity
#import scipy.spatial.distance.euclidean as euclidean_distance

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')

parser.add_argument('--task_name', type=str, default='handbags2shoes', help='Set data name')
#parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--top5_path', type=str, default='./top5/', help='Set the path the top5 images will be saved')
parser.add_argument('--load_iter', type=float, help='load iteration suffix')
parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--embedding_encoder', type=str, default='vgg16', help='choose among pre-trained alexnet/vgg{11, 13, 16, 19}/vgg{11, 13, 16, 19}bn/resnet{18, 34, 50, 101, 152}/squeezenet{1.0, 1.1}/densenet{121, 169, 201, 161}/inceptionv3 models.')
#parser.add_argument('--similarity_metric', type=str, default='cosine', help='choose among cosine/euclidean similarity metrics.')

# TODO (lpupp) remove. bad idea -- loads all of the nets
# txt2model = {'alexnet': models.alexnet(pretrained=True),
#              'vgg11': models.vgg11(pretrained=True),
#              'vgg13': models.vgg13(pretrained=True),
#              'vgg16': models.vgg16(pretrained=True),
#              'vgg19': models.vgg19(pretrained=True),
#              'vgg11bn': models.vgg11_bn(pretrained=True),
#              'vgg13bn': models.vgg13_bn(pretrained=True),
#              'vgg16bn': models.vgg16_bn(pretrained=True),
#              'vgg19bn': models.vgg19_bn(pretrained=True),
#              'resnet18': models.resnet18(pretrained=True),
#              'resnet34': models.resnet34(pretrained=True),
#              'resnet50': models.resnet50(pretrained=True),
#              'resnet101': models.resnet101(pretrained=True),
#              'resnet152': models.resnet152(pretrained=True),
#              'squeezenet1.0': models.squeezenet1_0(pretrained=True),
#              'squeezenet1.1': models.squeezenet1_1(pretrained=True),
#              'densenet121': models.densenet121(pretrained=True),
#              'densenet169': models.densenet169(pretrained=True),
#              'densenet161': models.densenet161(pretrained=True),
#              'densenet201': models.densenet201(pretrained=True),
#              'inceptionv3': models.inception_v3(pretrained=True)}

#txt2model = {'euclidean': cos_sim,
#             'cosine': euclid}


def main():

    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    print('cuda: {}'.format(cuda))

    # TODO (lpupp) do this in general
    #embedding_encoder, enc_input_size = initialize_model(args.embedding_encoder)
    print('loading model. embedding_encoder {}'.format(args.embedding_encoder))
    embedding_encoder = txt2model[args.embedding_encoder]
    set_parameter_requires_grad(embedding_encoder, feature_extracting=True)
    embedding_encoder = nn.Sequential(
            *list(embedding_encoder.features.children())[:-1],
            nn.MaxPool2d(kernel_size=14)
            )
    enc_input_size = 224

    if cuda:
        embedding_encoder = embedding_encoder.cuda()

    embedding_encoder.eval()

    model_path = os.path.join(args.model_path, args.task_name)
    model_path = os.path.join(model_path, args.model_arch + str(args.image_size))

    top5_path = os.path.join(args.top5_path, args.task_name)
    top5_path = os.path.join(top5_path, args.model_arch + str(args.image_size))
    if not os.path.exists(top5_path):
        os.makedirs(top5_path)

    print('model_path {}'.format(model_path))
    print('top5_path {}'.format(top5_path))

    _, _, test_style_A, test_style_B = get_data()

    print('Reading images')
    test_A = read_images(test_style_A, args.image_size)
    test_B = read_images(test_style_B, args.image_size)
    n_A_img, n_B_img = test_A.shape[0], test_B.shape[0]

    with torch.no_grad():
        test_A = Variable(torch.FloatTensor(test_A))
    with torch.no_grad():
        test_B = Variable(torch.FloatTensor(test_B))

    if cuda:
        test_A = test_A.cuda()
        test_B = test_B.cuda()

    print('Loading generator')
    ix = str(args.load_iter)
    generator_A = torch.load(os.path.join(model_path, 'model_gen_A-' + ix))
    generator_B = torch.load(os.path.join(model_path, 'model_gen_B-' + ix))

    # translate all images (A and B)
    print('Translating images ---------')
    print('A to B')
    AB = generator_B(test_A)
    print('B to A')
    BA = generator_A(test_B)

    # Up to here is fine!!!!!!!! ##############################################
    # #########################################################################
    # TODO(lpupp) Below here is not fine!!!!!!!! ##############################
    # The problem is the transform crap
    # Why do I need the transform stuff? because the encoder dimension expects
    # a different sized input.

    # Normalize all inputs to embedding_encoder
    dsize = (enc_input_size, enc_input_size)
    #normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                             std=[0.229, 0.224, 0.225])
    #tfs = torchvision.transforms.Compose([
    #        torchvision.transforms.ToPILImage(),
    #        torchvision.transforms.Resize(dsize, interpolation=2),
    #        torchvision.transforms.ToTensor(),
    #        normalize
    #        ])

    # TODO (lpupp) Is there a way around this song and dance?
    A_enc, B_enc, AB_enc, BA_enc = [], [], [], []
    A, B, AB, BA = as_np(test_A), as_np(test_B), as_np(AB), as_np(BA)

    print('Converting and resizing image tensors to numpy')
    for i in range(n_A_img):
        #A_enc.append(tfs(A[i].astype(np.uint8)))
        A_enc.append(cv2.resize(A[i].transpose(1, 2, 0), dsize=dsize, interpolation=cv2.INTER_CUBIC))
        AB_enc.append(cv2.resize(AB[i].transpose(1, 2, 0), dsize=dsize, interpolation=cv2.INTER_CUBIC))
    for i in range(n_B_img):
        #B_enc.append(tfs(B[i].astype(np.uint8)))
        B_enc.append(cv2.resize(B[i].transpose(1, 2, 0), dsize=dsize, interpolation=cv2.INTER_CUBIC))
        BA_enc.append(cv2.resize(BA[i].transpose(1, 2, 0), dsize=dsize, interpolation=cv2.INTER_CUBIC))

    A_enc = np.stack(A_enc).transpose(0, 3, 1, 2)
    B_enc = np.stack(B_enc).transpose(0, 3, 1, 2)
    AB_enc = np.stack(AB_enc).transpose(0, 3, 1, 2)
    BA_enc = np.stack(BA_enc).transpose(0, 3, 1, 2)

    with torch.no_grad():
        A_enc = Variable(torch.FloatTensor(A_enc))
    with torch.no_grad():
        B_enc = Variable(torch.FloatTensor(B_enc))
    with torch.no_grad():
        AB_enc = Variable(torch.FloatTensor(AB_enc))
    with torch.no_grad():
        BA_enc = Variable(torch.FloatTensor(BA_enc))

    # TODO (lpupp) batch
    # TODO (lpupp) im getting pretty weird outputs
    # Encode all translated images (A, B, AB and BA)
    print('Encoding images --------------')
    print('A')
    A_encoded = embedding_encoder(A_enc)
    print('B')
    B_encoded = embedding_encoder(B_enc)
    print('AB')
    AB_encoded = embedding_encoder(AB_enc)
    print('BA')
    BA_encoded = embedding_encoder(BA_enc)
    # TODO (lpupp) Could output this to csv...

    # #########################################################################
    # #########################################################################
    # #########################################################################
    # #########################################################################
    # Below here should be fine but needs to be tested

    # For each translation (AB and BA) find top 5 similarity (in B and A resp.)
    print('find_top_n_similar --------------')
    AB_similar = find_top_n_similar(AB_encoded, B_encoded, n=5)
    BA_similar = find_top_n_similar(BA_encoded, A_encoded, n=5)

    # Plot results nicely
    print('plot --------------')
    plot_all_outputs(AB_similar, [A, AB, B], src_style='A', path=top5_path)
    plot_all_outputs(BA_similar, [B, BA, A], src_style='B', path=top5_path)
