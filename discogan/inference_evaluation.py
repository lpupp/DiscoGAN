import os
import argparse
import math
from itertools import chain, product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

#from dataset import *
from model import *
from utils import *
from data_utils import *

import scipy

#import sklearn.metrics.pairwise.cosine_similarity as cosine_similarity
#import scipy.spatial.distance.euclidean as euclidean_distance

#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')

parser.add_argument('--domain', type=str, default='fashion', help='Set data domain. Choose among `fashion` or `furniture`')
parser.add_argument('--data_A', type=str, default=None, help='Set source data domain. if domain==`fashion`, choose among [`handbags`, `shoes`, `belts`, `dresses`]; if domain==`furniture`, choose among choose among [`seating`, `tables`, `storage`].')
# TODO(lpupp) remove task_name
parser.add_argument('--task_name', type=str, default='shoes2handbags', help='Set data name')

#parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--topn_path', type=str, default='./top5/', help='Set the path the top5 images will be saved')
parser.add_argument('--load_iter', type=float, help='load iteration suffix')
parser.add_argument('--topn', type=int, defaults=5, help='load iteration suffix')
parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--embedding_encoder', type=str, default='vgg16', help='choose among pre-trained alexnet/vgg{11, 13, 16, 19}/vgg{11, 13, 16, 19}bn/resnet{18, 34, 50, 101, 152}/squeezenet{1.0, 1.1}/densenet{121, 169, 201, 161}/inceptionv3 models')
#parser.add_argument('--similarity_metric', type=str, default='cosine', help='choose among cosine/euclidean similarity metrics.')

parser.add_argument('--seed', type=int, default=0, help='Set seed')

# TODO (lpupp) remove. bad idea -- loads all of the nets
#txt2model = {'alexnet': models.alexnet(pretrained=True),
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

domain_l = {'furniture': ['seating', 'tables', 'storage'],
            'fashion': ['handbags', 'shoes', 'belts', 'dresses']}
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def create_model_path(task_name):
    # TODO(lpupp) test
    model_path = os.path.join(args.model_path, task_name)
    model_path = os.path.join(model_path, args.model_arch + str(args.image_size))
    return model_path


def create_nms(task_name, domain2label):
    # TODO(lpupp) test
    A, B = task_name.split('2')
    return domain2label[B] + domain2label[A], domain2label[A] + domain2label[B]


def main():

    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False

    print('cuda: {}'.format(cuda))

    use_all_domains = args.data_A is None

    # TODO(lpupp)
    _tmp = domain_l[args.domain]
    data_A = args.data_A or _tmp[0]
    _tmp.insert(0, _tmp.pop(_tmp.index(data_A)))
    domain_labs = letters[:len(_tmp)]
    domain = dict((k, v) for k, v in zip(domain_labs, _tmp))
    domain2lab = dict((k, v) for k, v in zip(_tmp, letters[:len(_tmp)]))

    label_perms = [i + j for i, j in product(domain_labs, domain_labs) if i != j]

    # TODO (lpupp) Do this in general
    #embedding_encoder, enc_input_size = initialize_model(args.embedding_encoder)
    print('loading model. embedding_encoder {}'.format(args.embedding_encoder))
    embedding_encoder = models.vgg19(pretrained=True)
    #embedding_encoder = txt2model[args.embedding_encoder]
    set_parameter_requires_grad(embedding_encoder, feature_extracting=True)
    embedding_encoder = nn.Sequential(
            *list(embedding_encoder.features.children())[:-1],
            nn.MaxPool2d(kernel_size=14)
            )
    enc_input_size = 224

    if cuda:
        embedding_encoder = embedding_encoder.cuda()

    embedding_encoder.eval()

    #model_path = os.path.join(args.model_path, args.task_name)
    #model_path = os.path.join(model_path, args.model_arch + str(args.image_size))

    # TODO(lpupp) this will not work with args.task_name
    topn_path = os.path.join(args.topn_path, args.task_name)
    topn_path = os.path.join(topn_path, args.model_arch + str(args.image_size))
    if not os.path.exists(topn_path):
        os.makedirs(topn_path)

    print('model_path {}'.format(model_path))
    print('topn_path {}'.format(topn_path))

    img_paths = dict((k, get_photo_files(v)) for k, v in domain.items())
    #_, _, test_style_A, test_style_B = get_data(args)

    print('Reading images')
    imgs = dict((k, read_images(v, args.image_size)) for k, v in img_paths.items())
    n_imgs = dict((k, v.shape[0]) for k, v in test_imgs.items())
    #test_A = read_images(test_style_A, args.image_size)
    #test_B = read_images(test_style_B, args.image_size)
    #n_A_img, n_B_img = test_A.shape[0], test_B.shape[0]

    for k in imgs:
        with torch.no_grad():
            imgs[k] = Variable(torch.FloatTensor(imgs[k]))
        if cuda:
            imgs[k] = imgs[k].cuda()

    #with torch.no_grad():
    #    test_A = Variable(torch.FloatTensor(test_A))
    #with torch.no_grad():
    #    test_B = Variable(torch.FloatTensor(test_B))
    #
    #if cuda:
    #    test_A = test_A.cuda()
    #    test_B = test_B.cuda()

    print('Loading generator ------------------------------------------------')

    ix = str(args.load_iter)

    generators = {}
    task_names = os.listdir(args.model_path)
    # TODO(lpupp) test
    for nm in task_names:
        path = create_model_path(nm)
        A_nm, B_nm = create_nms(nm, domain2lab)
        generators[A_nm] = torch.load(os.path.join(path, 'model_gen_A-' + ix))
        generators[B_nm] = torch.load(os.path.join(path, 'model_gen_B-' + ix))

    #model_path = os.path.join(args.model_path, get_task_name(A, B, model_paths))
    #model_path = os.path.join(model_path, args.model_arch + str(args.image_size))

    #generator_A = torch.load(os.path.join(model_path, 'model_gen_A-' + ix))
    #generator_B = torch.load(os.path.join(model_path, 'model_gen_B-' + ix))

    # translate all images (A and B)
    print('Translating images -----------------------------------------------')
    for lab_perm in label_perms:
        # TODO(lpupp) test
        print('{} to {}'.format(lab_perm[0], lab_perm[1]))
        imgs[lab_perm] = generators[lab_perm[1]](imgs[lab_perm[0]])

    #print('A to B')
    #AB = generator_B(test_A)
    #print('B to A')
    #BA = generator_A(test_B)

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

    def resize_img(x, dsize):
        return cv2.resize(x.transpose(1, 2, 0), dsize=dsize, interpolation=cv2.INTER_CUBIC)

    # TODO (lpupp) Is there a way around this song and dance?
    #A_enc, B_enc, AB_enc, BA_enc = [], [], [], []
    imgs_np = dict((k, as_np(v)) for k, v in imgs.items())
    #A, B, AB, BA = as_np(test_A), as_np(test_B), as_np(AB), as_np(BA)

    print('Converting and resizing image tensors to numpy')
    # TODO(lpupp) Does this work?
    imgs_enc = dict((k, [resize_img(i, dsize) for i in v]) for k, v in imgs_np.items())
    #for i in range(n_A_img):
    #    #A_enc.append(tfs(A[i].astype(np.uint8)))
    #    A_enc.append(resize_img(A[i], dsize))
    #    AB_enc.append(resize_img(AB[i], dsize))
    #for i in range(n_B_img):
    #    #B_enc.append(tfs(B[i].astype(np.uint8)))
    #    B_enc.append(resize_img(B[i], dsize))
    #    BA_enc.append(resize_img(BA[i], dsize))

    # TODO(lpupp) test
    imgs_enc = dictionary_map(imgs_enc, lambda x: np.stack(x).transpose(0, 3, 1, 2))
    #A_enc = np.stack(A_enc).transpose(0, 3, 1, 2)
    #B_enc = np.stack(B_enc).transpose(0, 3, 1, 2)
    #AB_enc = np.stack(AB_enc).transpose(0, 3, 1, 2)
    #BA_enc = np.stack(BA_enc).transpose(0, 3, 1, 2)

    for k in imgs_enc:
        with torch.no_grad():
            imgs_enc[k] = Variable(torch.FloatTensor(imgs_enc[k]))
        if cuda:
            imgs_enc[k] = imgs_enc[k].cuda()

    #with torch.no_grad():
    #    A_enc = Variable(torch.FloatTensor(A_enc))
    #with torch.no_grad():
    #    B_enc = Variable(torch.FloatTensor(B_enc))
    #with torch.no_grad():
    #    AB_enc = Variable(torch.FloatTensor(AB_enc))
    #with torch.no_grad():
    #    BA_enc = Variable(torch.FloatTensor(BA_enc))
    #
    #if cuda:
    #    A_enc = A_enc.cuda()
    #    B_enc = B_enc.cuda()
    #    AB_enc = AB_enc.cuda()
    #    BA_enc = BA_enc.cuda()

    # TODO (lpupp) batch
    # TODO (lpupp) im getting pretty weird outputs
    # Encode all translated images (A, B, AB and BA)
    print('Encoding images --------------')
    # TODO(lpupp) test
    imgs_enc = dictionary_map(imgs_enc, lambda x: minibatch_call(x, embedding_encoder))
    #print('A')
    #A_encoded = minibatch_call(A_enc, embedding_encoder)
    #print(A_encoded.shape)
    #print('B')
    #B_encoded = minibatch_call(B_enc, embedding_encoder)
    #print(B_encoded.shape)
    #print('AB')
    #AB_encoded = minibatch_call(AB_enc, embedding_encoder)
    #print(AB_encoded.shape)
    #print('BA')
    #BA_encoded = minibatch_call(BA_enc, embedding_encoder)
    #print(BA_encoded.shape)
    # TODO (lpupp) Could output this to csv...

    # #########################################################################
    # #########################################################################
    # #########################################################################
    # #########################################################################
    # Below here should be fine but needs to be tested

    # For each translation (AB and BA) find top 5 similarity (in B and A resp.)
    sim_disco, sim_vgg = {}, {}
    # TODO(lpupp) test
    print('find top n similar (discoGAN) ------------------------------------')
    for lab_perm in label_perms:
        print(lab_perm)
        sim_disco[lab_perm] = find_top_n_similar(imgs_enc[lab_perm], imgs_enc[lab_perm[1]], n=args.topn)

    print('find top n similar (VGG) -----------------------------------------')
    for lab in domain_labs:
        remaining_labs = list(filter(lambda x: x != lab, domain_labs))
        for i in remaining_labs:
            sim_vgg[lab + '2' + i] = find_top_n_similar(imgs_enc[i], imgs_enc[lab], n=args.topn)

    #print('AB')
    #AB_similar = find_top_n_similar(AB_encoded, B_encoded, n=args.topn)
    #print('BA')
    #BA_similar = find_top_n_similar(BA_encoded, A_encoded, n=args.topn)

    # Plot results nicely
    # TODO(lpupp) adjust for multiple categories
    print('plot --------------')
    print('AB')
    plot_all_outputs(AB_similar, [A, AB, B], src_style='A', path=topn_path)
    print('BA')
    plot_all_outputs(BA_similar, [B, BA, A], src_style='B', path=topn_path)

    # TODO(lpupp) Do real-world run through with a zalando image. Compare our
    #             output with their output. This would require a harvesting of
    #             the zalando database for a (at least) a few products...
    # TODO(lpupp) Compare performance to discoGAN encoders without decoders
    # TODO(lpupp) Update existing script to return top n images from any class


if __name__ == "__main__":
    main()
