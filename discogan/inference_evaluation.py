import os
os.chdir('/Users/lucagaegauf/Documents/GitHub/DiscoGAN')
import sys
sys.path.append('./discogan/')

import argparse
from itertools import product

import torch
import torch.nn as nn
import torchvision.models as models

from model import *
from utils import *
from data_utils import *

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--cuda', type=bool, default=False, help='Set cuda usage')

parser.add_argument('--domain', type=str, default='fashion', help='Set data domain. Choose among `fashion` or `furniture`')
parser.add_argument('--topn', type=int, default=5, help='load iteration suffix')

#parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_path', type=str, default='/Users/lucagaegauf/Dropbox/GAN/models/', help='Set the path for trained models')
parser.add_argument('--topn_path', type=str, default='./top5/', help='Set the path the top5 images will be saved')

parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--embedding_encoder', type=str, default='vgg19', help='choose among pre-trained alexnet/vgg{11, 13, 16, 19}/vgg{11, 13, 16, 19}bn/resnet{18, 34, 50, 101, 152}/squeezenet{1.0, 1.1}/densenet{121, 169, 201, 161}/inceptionv3 models')
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

domain_d = {'furniture': ['seating', 'tables', 'storage'],
            'fashion': ['handbags', 'shoes', 'belts', 'dresses']}
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def create_nms(task_name, domain2label):
    A, B = task_name.split('2')
    return domain2label[B] + domain2label[A], domain2label[A] + domain2label[B]


def main():
    """Main TODO.

    Steps:
    - Load data.
    - Load generators.
    - Translate all images with generators.
    - Encode images with pretrained encoder.
    - Get top n.
    - Plot everything.

    TODO(lpupp)
    [ ] Do real-world run through with a zalando image. Compare our output with
        their output. This would require a harvesting of the zalando database
        for a (at least) a few products...
    [ ] Compare topn with and without source class
    [ ] Update existing script to return top n images from any class
    """

    global args
    args = parser.parse_args()

    cuda = args.cuda

    print('cuda: {}'.format(cuda))

    #use_all_domains = args.data_A is None

    model_arch = args.model_arch + str(args.image_size)

    # TODO(lpupp)
    d_nm = args.domain
    domain_set = domain_d[d_nm]
    #data_A = args.data_A or domain_set[0]
    #domain_set.insert(0, domain_set.pop(domain_set.index(data_A)))
    domain_labs = letters[:len(domain_set)]

    domain = dict((k, v) for k, v in zip(domain_labs, domain_set))
    domain2lab = dict((v, k) for k, v in domain.items())

    label_perms = [i + j for i, j in product(domain_labs, domain_labs) if i != j]

    # TODO (lpupp) Do this in general
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

    # TODO(lpupp)
    for nm in domain.values():
        topn_path = os.path.join(args.topn_path, d_nm, nm, model_arch)
        if not os.path.exists(topn_path):
            os.makedirs(topn_path)
        topn_path = os.path.join(args.topn_path, d_nm, nm, 'vgg')
        if not os.path.exists(topn_path):
            os.makedirs(topn_path)

    img_paths = dict_map(domain, lambda v: get_photo_files(v)[1])

    print('Reading images ---------------------------------------------------')
    imgs = dict_map(img_paths, lambda v: read_images(v, args.image_size))
    # TODO(lpupp) for some reason there are only 399 seating loaded...
    n_imgs = dict_map(imgs, lambda v: v.shape[0])
    imgs = dict_map(imgs, lambda v: torch_cuda(v, cuda))

    print('Loading generator ------------------------------------------------')
    generators = {}
    task_names = [e for e in os.listdir(args.model_path) if '2' in e]
    task_names = [e for e in task_names if domain['A'] in e or domain['B'] in e]

    for i, nm in enumerate(task_names):
        path = os.path.join(args.model_path, d_nm, nm, model_arch)
        ix = max([float(e.split('-')[1]) for e in os.listdir(path) if 'model_gen' in e])
        A_nm, B_nm = create_nms(nm, domain2lab)
        generators[A_nm] = torch.load(os.path.join(path, 'model_gen_A-' + str(ix)))#, map_location={'cuda:0': 'cpu'})
        generators[B_nm] = torch.load(os.path.join(path, 'model_gen_B-' + str(ix)))#, map_location={'cuda:0': 'cpu'})

    # translate all images (A and B)
    print('Translating images -----------------------------------------------')
    for ab in label_perms:
        print('{} to {}'.format(ab[0], ab[1]))
        imgs[ab] = generators[ab](imgs[ab[0]])

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
    print('Converting and resizing image tensors to numpy -------------------')
    imgs_np = dict_map(imgs, lambda v: as_np(v))
    imgs_enc = dict_map(imgs_np, lambda v: resize_array_of_images(v, dsize))
    imgs_enc = dict_map(imgs_enc, lambda v: torch_cuda(v, cuda))

    # Encode all translated images (A, B, AB and BA)
    print('Encoding images --------------------------------------------------')
    imgs_enc = dict_map(imgs_enc, lambda v: minibatch_call(v, embedding_encoder))
    print(dict_map(imgs_enc, lambda v: v.shape))
    # TODO (lpupp) Could output this to csv...

    # #########################################################################
    # #########################################################################
    # #########################################################################
    # #########################################################################
    # Below here should be fine but needs to be tested

    print('Find top n similar -----------------------------------------------')
    print('Using discoGAN')
    # For each translation (AB and BA) find top n similarity (in B and A resp.)
    sim_disco, sim_vgg = {}, {}
    for ab in label_perms:
        sim_disco[ab] = find_top_n_similar_dict(imgs_enc[ab], imgs_enc[ab[1]], n=args.topn)

    print('Using pretrained {}'.format(args.embedding_encoder))
    for a in domain_labs:
        for b in domain_labs:
            if a == b:
                tmp = find_top_n_similar_dict(imgs_enc[b], imgs_enc[a], n=args.topn+1)
                sim_vgg[a+b] = dict_map(tmp, lambda v: v[:-1])
            else:
                sim_vgg[a+b] = find_top_n_similar_dict(imgs_enc[b], imgs_enc[a], n=args.topn)

    # Plot results nicely
    print('Ploting results --------------------------------------------------')
    # Plot top n similar using discogan results
    for ab in label_perms:
        print(ab)
        a, b = ab[0], ab[1]
        plot_all_outputs(sim_disco[ab],
                         [imgs_np[a], imgs_np[ab], imgs_np[b]],
                         src_style=str(ab),
                         path=os.path.join(args.topn_path, d_nm, domain[a], model_arch))

    # Plot top n similar using VGG results
    for ab in sim_vgg:
        print(ab)
        a, b = ab[0], ab[1]
        plot_all_outputs(sim_vgg[ab],
                         [imgs_np[a], np.ones_like(imgs_np[a]), imgs_np[b]],
                         src_style=str(ab),
                         path=os.path.join(args.topn_path, d_nm, domain[a], 'vgg'))


def transform_single_image(img_path, img_class, topn):
    """transform_single_image TODO.

    Steps:
    - Load data.
    - Load generators.
    - Translate all images with generators.
    - Encode images with pretrained encoder.
    - Get top n.
    - Plot everything.
    """

    #img_path = '/Users/lucagaegauf/Dropbox/GAN/fashion/shoes/val/1_AB.jpg'
    #img_class = 'shoes'
    global args
    args = parser.parse_args()

    cuda = args.cuda

    print('cuda: {}'.format(cuda))

    model_arch = args.model_arch + str(args.image_size)

    d_nm = args.domain
    domain_set = domain_d[d_nm]
    domain_set.insert(0, domain_set.pop(domain_set.index(img_class)))
    domain_labs = letters[:len(domain_set)]

    domain = dict((k, v) for k, v in zip(domain_labs, domain_set))
    domain2lab = dict((v, k) for k, v in domain.items())

    # TODO(lpupp) Do this in general
    # TODO(lpupp) stick this in if __name__ == "__main__"
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

    # TODO(lpupp) can i do this in main() too?
    model_path = os.path.join(args.model_path, d_nm)
    topn_path = os.path.join(args.topn_path, 'single_img', model_arch)
    if not os.path.exists(topn_path):
        os.makedirs(topn_path)

    print('Reading images ---------------------------------------------------')
    source_img_np = read_image(img_path, args.image_size)
    source_img_np = np.expand_dims(source_img_np, 0)
    source_img = torch_cuda(source_img_np, cuda)

    img_paths = dict_map(domain, lambda v: get_photo_files(v)[1])
    img_paths['A'].remove(img_path)
    imgs_np = dict_map(img_paths, lambda v: read_images(v, args.image_size))

    print('Loading generator ------------------------------------------------')
    generators = {}
    task_names = [e for e in os.listdir(model_path) if img_class in e]

    # TODO(lpupp) make generators keys AB, AC, AD ...
    for i, nm in enumerate(task_names):
        path = os.path.join(model_path, nm, model_arch)
        A, B = nm.split('2')
        ix = max([float(e.split('-')[1]) for e in os.listdir(path) if 'model_gen' in e])
        domain2lab
        if A == img_class:
            nm = 'A' + domain2lab[B]
            generators[nm] = torch.load(os.path.join(path, 'model_gen_B-' + str(ix)), map_location={'cuda:0': 'cpu'})
        else:
            nm = 'A' + domain2lab[A]
            generators[nm] = torch.load(os.path.join(path, 'model_gen_A-' + str(ix)), map_location={'cuda:0': 'cpu'})

    # translate all images (A and B)
    print('Translating source image -----------------------------------------')
    # TODO(lpupp) does this work????
    img_trans = dict_map(generators, lambda v: v(source_img))

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
    print('Converting and resizing image tensors to numpy -------------------')
    img_trans_np = dict_map(img_trans, lambda v: as_np(v))

    source_img_enc = resize_array_of_images(source_img_np, dsize)
    img_trans_enc = dict_map(img_trans_np, lambda v: resize_array_of_images(v, dsize))
    imgs_enc = dict_map(imgs_np, lambda v: resize_array_of_images(v, dsize))

    source_img_enc = torch_cuda(source_img_enc, cuda)
    img_trans_enc = dict_map(img_trans_enc, lambda v: torch_cuda(v, cuda))
    imgs_enc = dict_map(imgs_enc, lambda v: torch_cuda(v, cuda))

    # Encode all translated images (A, B, AB and BA)
    print('Encoding images --------------------------------------------------')
    source_img_enc = embedding_encoder(source_img_enc)
    img_trans_enc = dict_map(img_trans_enc, lambda v: embedding_encoder(v))
    imgs_enc = dict_map(imgs_enc, lambda v: minibatch_call(v, embedding_encoder))

    # #########################################################################
    # #########################################################################
    # #########################################################################
    # #########################################################################
    # Below here should be fine but needs to be tested

    print('Find top n similar using discoGAN --------------------------------')
    # For each translation (AB and BA) find top n similarity (in B and A resp.)
    sim = {'A': find_top_n_similar_by_img_dict(torch.squeeze(source_img_enc), imgs_enc['A'], n=topn)}
    for k in img_trans_enc:
        b = k[1]
        sim[b] = find_top_n_similar_by_img_dict(torch.squeeze(img_trans_enc[k]), imgs_enc[b], n=topn)

    sim_total = list((k, e) for k, v in sim.items() for e in v)
    topn_overall = sorted(sim_total, key=lambda kv: kv[1][1])[-topn:]
    
    # Plot results nicely
    print('Ploting results --------------------------------------------------')
    # Plot top n similar using discogan results
    img_trans_np['AA'] = np.ones_like(source_img_np)
    for k in sim:
        plot_outputs(0,
                     sim[k],
                     [source_img_np, img_trans_np['A'+k], imgs_np[k]],
                     src_style=str(img_class+'2'+domain[k]),
                     path=topn_path)
            
    del img_trans_np['AA']
    plot_overall(topn_overall, source_img_np, img_trans_np, imgs_np, img_ix=0, path=topn_path)

if __name__ == "__main__":
    main()
    #transform_single_image(img_path='/Users/lucagaegauf/Dropbox/GAN/fashion/shoes/val/1_AB.jpg',
    #                       img_class='shoes',
    #                       topn=args.topn)

