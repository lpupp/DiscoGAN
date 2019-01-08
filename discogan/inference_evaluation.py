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

    # TODO (lpupp) do this in general
    #embedding_encoder, enc_input_size = initialize_model(args.embedding_encoder)

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

    _, _, test_style_A, test_style_B = get_data()

    test_A = read_images(test_style_A, args.image_size)
    test_B = read_images(test_style_B, args.image_size)

    with torch.no_grad():
        test_A = Variable(torch.FloatTensor(test_A))
    with torch.no_grad():
        test_B = Variable(torch.FloatTensor(test_B))

    if cuda:
        test_A = test_A.cuda()
        test_B = test_B.cuda()

    ix = str(args.load_iter)
    generator_A = torch.load(os.path.join(model_path, 'model_gen_A-' + ix))
    generator_B = torch.load(os.path.join(model_path, 'model_gen_B-' + ix))

    # translate all images (A and B)
    AB = generator_B(test_A)
    BA = generator_A(test_B)

    # Up to here is fine!!!!!!!! ##############################################
    # #########################################################################
    # TODO(lpupp) Below here is not fine!!!!!!!! ##############################

    # Normalize all inputs to embedding_encoder
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((enc_input_size, enc_input_size), interpolation=2),
            torchvision.transforms.ToTensor(),
            normalize
            ])

    # TODO (lpupp) Is there a way around this song and dance?
    A_, B_, AB_, BA_ = [], [], [], []
    AB, BA = as_np(AB), (BA)
    A, B, AB, BA = as_np(test_A), as_np(test_B), as_np(AB), as_np(BA)
    for i in range(A.shape[0]):
        A_.append(tfs(A[i].astype(np.uint8)))
    for i in range(B.shape[0]):
        B_.append(tfs(B[i].astype(np.uint8)))
    for i in range(AB.shape[0]):
        AB_.append(tfs(AB[i].astype(np.uint8)))
    for i in range(BA.shape[0]):
        BA_.append(tfs(BA[i].astype(np.uint8)))

    A = np.stack(A)
    B = np.stack(B)
    AB = np.stack(AB_)
    BA = np.stack(BA_)

    with torch.no_grad():
        A = Variable(torch.FloatTensor(A))
    with torch.no_grad():
        B = Variable(torch.FloatTensor(B))
    with torch.no_grad():
        AB = Variable(torch.FloatTensor(AB))
    with torch.no_grad():
        BA = Variable(torch.FloatTensor(BA))

    # TODO (lpupp) batch
    # TODO (lpupp) im getting pretty weird outputs
    # Encode all translated images (A, B, AB and BA)
    A_encoded = embedding_encoder(A)
    B_encoded = embedding_encoder(B)
    AB_encoded = embedding_encoder(AB)
    BA_encoded = embedding_encoder(BA)
    # TODO (lpupp) Could output this to csv...

    # #########################################################################
    # #########################################################################
    # #########################################################################
    # #########################################################################
    # Below here should be fine but needs to be tested

    # For each translation (AB and BA) find top 5 similarity (in B and A resp.)
    AB_similar = find_top_n_similar(AB_encoded, B_encoded, n=5)
    BA_similar = find_top_n_similar(BA_encoded, A_encoded, n=5)

    # Plot results nicely
    plot_all_outputs(AB_similar, src_style='A', path=top5_path)
    plot_all_outputs(BA_similar, src_style='B', path=top5_path)
