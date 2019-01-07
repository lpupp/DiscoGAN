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
import scipy

from sklearn.metrics import pairwise.cosine_similarity
from scipy.spatial import distance.euclidean

import matplotlib.pyplot as plt

#from progressbar import ETA, Bar, Percentage, ProgressBar

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')

parser.add_argument('--task_name', type=str, default='handbags2shoes', help='Set data name')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--load_iter', type=float, help='load iteration suffix')
parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--embedding_encoder', type=str, default='vgg16', help='choose among pre-trained alexnet/vgg{11, 13, 16, 19}/vgg{11, 13, 16, 19}bn/resnet{18, 34, 50, 101, 152}/squeezenet{1.0, 1.1}/densenet{121, 169, 201, 161}/inceptionv3 models.')
#parser.add_argument('--similarity_metric', type=str, default='cosine', help='choose among cosine/euclidean similarity metrics.')

# Import these from some utils file
#def cos_sim(x, y):
#    return pairwise.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).item()

#def euclid(x, y):
#    return distance.euclidean(x, y)

txt2model = {'alexnet': models.alexnet(pretrained=True),
             'vgg11': models.vgg11(pretrained=True),
             'vgg13': models.vgg13(pretrained=True),
             'vgg16': models.vgg16(pretrained=True),
             'vgg19': models.vgg19(pretrained=True),
             'vgg11bn': models.vgg11_bn(pretrained=True),
             'vgg13bn': models.vgg13_bn(pretrained=True),
             'vgg16bn': models.vgg16_bn(pretrained=True),
             'vgg19bn': models.vgg19_bn(pretrained=True),
             'resnet18': models.resnet18(pretrained=True),
             'resnet34': models.resnet34(pretrained=True),
             'resnet50': models.resnet50(pretrained=True),
             'resnet101': models.resnet101(pretrained=True),
             'resnet152': models.resnet152(pretrained=True),
             'squeezenet1.0': models.squeezenet1_0(pretrained=True),
             'squeezenet1.1': models.squeezenet1_1(pretrained=True),
             'densenet121': models.densenet121(pretrained=True),
             'densenet169': models.densenet169(pretrained=True),
             'densenet161': models.densenet161(pretrained=True),
             'densenet201': models.densenet201(pretrained=True),
             'inceptionv3': models.inception_v3(pretrained=True)}

#txt2model = {'euclidean': cos_sim,
#             'cosine': euclid}


def find_top_n_similar(source_embeds, db_embeds, n=1):
    out = {}
    for i in range(source_embeds.shape[0]):
        out[i] = find_top_n_similar_by_img(torch.squeeze(source_embeds[i]), db_embeds, n=n)
    return out


def find_top_n_similar_by_img(embed, db_embeds, n=1):
    sim = []
    for i in range(db_embeds.shape[0]):
        sim.append(as_np(F.cosine_similarity(embed, torch.squeeze(db_embeds[i]), dim=0)).item())
    return sorted(list(range(len(sim))), key=lambda i: sim[i])[-n:]


def as_np(data):
    return data.cpu().data.numpy()


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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def main():

    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    embedding_encoder = txt2model[args.embedding_encoder]
    # TODO (lpupp) How to do this in general?
    set_parameter_requires_grad(embedding_encoder, feature_extracting=True)
    embedding_encoder = nn.Sequential(
            *list(embedding_encoder.features.children())[:-1],
            nn.MaxPool2d(kernel_size=14)
            )

enc_input_size = 224

if cuda:
    embedding_encoder = embedding_encoder.cuda()
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print('device: {}'.format(device))
#embedding_encoder = embedding_encoder.to(device)

embedding_encoder.eval()


    model_path = os.path.join(args.model_path, args.task_name)
    model_path = os.path.join(model_path, args.model_arch + str(args.image_size))

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

    # TODO (@lpupp)
    subdir_path = os.path.join(result_path, str(iters / args.image_save_interval))

    if os.path.exists(subdir_path):
        pass
    else:
        os.makedirs(subdir_path)

    # n_testset = min(test_A.size()[0], test_B.size()[0])
    # for im_idx in range(n_testset):
    #
    #     #A_val = test_A[im_idx].cpu().data.numpy().transpose(1,2,0) * 255.
    #     #B_val = test_B[im_idx].cpu().data.numpy().transpose(1,2,0) * 255.
    #     #BA_val = BA[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
    #     #AB_val = AB[im_idx].cpu().data.numpy().transpose(1,2,0)* 255.
    #
    #     filename_prefix = os.path.join(subdir_path, str(im_idx))
    #     img4save(test_A[im_idx])
    #     scipy.misc.imsave(filename_prefix + '.A.jpg', img4save(test_A[im_idx]))
    #     scipy.misc.imsave(filename_prefix + '.B.jpg', img4save(test_B[im_idx]))
    #     scipy.misc.imsave(filename_prefix + '.BA.jpg', img4save(BA[im_idx]))
    #     scipy.misc.imsave(filename_prefix + '.AB.jpg', img4save(AB[im_idx]))

    # translate all images (A and B)
    AB = generator_B(test_A)
    BA = generator_A(test_B)

    # Normalize all inputs to embedding_encoder
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    tfs = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((input_size, input_size), interpolation=2),
            torchvision.transforms.ToTensor(),
            normalize
            ])

    # TODO (lpupp) Is there a way around this song and dance?
    A_, B_, AB_, BA_ = [], [],[], []
    AB, BA = as_np(AB), as_np(BA)
    for i in range(test_A.shape[0]):
        A_.append(tfs(test_A[i].astype(np.uint8)))
    for i in range(test_A.shape[0]):
        B_.append(tfs(test_B[i].astype(np.uint8)))
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

    # Encode all translated images (A, B, AB and BA)
    A_encoded = embedding_encoder(A)
    B_encoded = embedding_encoder(B)
    AB_encoded = embedding_encoder(AB)
    BA_encoded = embedding_encoder(BA)
    # TODO (lpupp) Could output this to csv...

    # For each translation (AB and BA) find top 5 similarity (in B and A resp.)
    AB_similar = find_top_n_similar(AB_encoded, B_encoded, n=5)
    BA_similar = find_top_n_similar(BA_encoded, A_encoded, n=5)

    # Plot results nicely
    # TODO (lpupp) Idk how to select images yet
