import os

import argparse
import time
import random
from itertools import product

import torch
import torch.nn as nn
import torchvision.models as models

from model import *
from utils import *
from data_utils import *

def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
add_bool_arg(parser, 'cuda')

parser.add_argument('--domain', type=str, default='fashion', help='Set data domain. Choose among `fashion` or `furniture`')
parser.add_argument('--topn', type=int, default=5, help='load iteration suffix')

parser.add_argument('--model_path', type=str, default='./final_models/', help='Set the path for trained models')
parser.add_argument('--topn_path', type=str, default='./top5/', help='Set the path the topn images will be saved')

parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')
parser.add_argument('--model_arch', type=str, default='discogan', help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--embedding_encoder', type=str, default='vgg19', help='choose among pre-trained alexnet/vgg{11, 13, 16, 19}/vgg{11, 13, 16, 19}bn/resnet{18, 34, 50, 101, 152}/squeezenet{1.0, 1.1}/densenet{121, 169, 201, 161}/inceptionv3 models')
#parser.add_argument('--similarity_metric', type=str, default='cosine', help='choose among cosine/euclidean similarity metrics.')

parser.add_argument('--image_path', type=str, default=None, help='If provided, single_image will be execute, else main. Path to image.')
parser.add_argument('--image_class', type=str, default=None, help='Class to with image_path image belongs. E.g. shoes')

parser.add_argument('--eval_task', type=str, default='out', help='Evaluate on `in` or `out`-of-sample data, `random` for random, or `single` for single image eval.')
parser.add_argument('--seed', type=int, default=0, help='Random seed')

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


def train_val_photos(v):
    train, val = get_photo_files(v)
    return train + val


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

    dsize = (enc_img_size, enc_img_size)

    for nm in domain.values():
        topn_path = os.path.join(paths['topn'], nm, model_arch, '_out')
        if not os.path.exists(topn_path):
            os.makedirs(topn_path)
        topn_path = os.path.join(paths['topn'], nm, 'vgg_out')
        if not os.path.exists(topn_path):
            os.makedirs(topn_path)

    img_paths = dict_map(domain, lambda v: get_photo_files(v)[1])

    print('Reading images ---------------------------------------------------')
    imgs = dict_map(img_paths, lambda v: read_images(v, img_size))
    imgs = dict_map(imgs, lambda v: torch_cuda(v, cuda))

    print('Loading generator ------------------------------------------------')
    generators = {}
    task_names = [e for e in os.listdir(paths['model']) if '2' in e]

    for i, nm in enumerate(task_names):
        path = os.path.join(paths['model'], nm, model_arch)
        ix = max([float(e.split('-')[1]) for e in os.listdir(path) if 'model_gen' in e])
        A_nm, B_nm = create_nms(nm, domain2lab)
        generators[A_nm] = torch.load(os.path.join(path, 'model_gen_A-' + str(ix)), map_location={'cuda:0': 'cpu'})
        generators[B_nm] = torch.load(os.path.join(path, 'model_gen_B-' + str(ix)), map_location={'cuda:0': 'cpu'})

    # translate all images (A and B)
    print('Translating images -----------------------------------------------')
    for ab in label_perms:
        print('{} to {}'.format(ab[0], ab[1]))
        imgs[ab] = generators[ab](imgs[ab[0]])

    # #########################################################################
    # TODO(lpupp) Is there around this? #######################################
    # The problem is the transform crap
    # Why do I need the transform stuff? because the encoder dimension expects
    # a different sized input.

    # Normalize all inputs to embedding_encoder
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
    imgs_enc = dict_map(imgs_enc, lambda v: minibatch_call(v, encoder))
    print(dict_map(imgs_enc, lambda v: v.shape))
    # TODO (lpupp) Could output this to csv...

    # #########################################################################

    print('Find top n similar -----------------------------------------------')
    print('Using discoGAN')
    # For each translation (AB and BA) find top n similarity (in B and A resp.)
    sim_disco, sim_vgg = {}, {}
    for ab in label_perms:
        sim_disco[ab] = find_top_n_similar(imgs_enc[ab], imgs_enc[ab[1]], n=topn)

    print('Using pretrained {}'.format(args.embedding_encoder))
    for a in domain_labs:
        for b in domain_labs:
            if a == b:
                tmp = find_top_n_similar(imgs_enc[a], imgs_enc[b], n=topn+1)
                sim_vgg[a+b] = dict_map(tmp, lambda v: v[:-1])
            else:
                sim_vgg[a+b] = find_top_n_similar(imgs_enc[a], imgs_enc[b], n=topn)

    # Plot results nicely
    print('Plotting results -------------------------------------------------')
    # Plot top n similar using discogan results
    for ab in label_perms:
        print(ab)
        a, b = ab[0], ab[1]
        plot_all_outputs(sim_disco[ab],
                         [imgs_np[a], imgs_np[ab], imgs_np[b]],
                         src_style=str(ab),
                         path=os.path.join(paths['topn'], domain[a], model_arch + '_out'))

    # Plot top n similar using VGG results
    for ab in sim_vgg:
        print(ab)
        a, b = ab[0], ab[1]
        plot_all_outputs(sim_vgg[ab],
                         [imgs_np[a], np.ones_like(imgs_np[a]), imgs_np[b]],
                         src_style=str(ab),
                         path=os.path.join(paths['topn'], domain[a], 'vgg_out'))


def eval_full_domain_set_in(cuda, encoder, model_arch, img_size, topn, domain, paths, enc_img_size):
    """TODO."""

    d_nm = domain
    domain_set = domain_d[d_nm]
    domain_labs = letters[:len(domain_set)]

    domain = dict((k, v) for k, v in zip(domain_labs, domain_set))
    domain2lab = dict((v, k) for k, v in domain.items())

    label_perms = [i + j for i, j in product(domain_labs, domain_labs) if i != j]

    dsize = (enc_img_size, enc_img_size)

    for nm in domain.values():
        topn_path = os.path.join(paths['topn'], nm, model_arch + '_in')
        if not os.path.exists(topn_path):
            os.makedirs(topn_path)
        #topn_path = os.path.join(paths['topn'], nm, 'vgg_in')
        #if not os.path.exists(topn_path):
        #    os.makedirs(topn_path)

    all_img_paths = dict_map(domain, lambda v: train_val_photos(v))
    img_paths = dict_map(domain, lambda v: get_photo_files(v)[1])

    print('Reading images ---------------------------------------------------')
    imgs = dict_map(img_paths, lambda v: read_images(v, img_size))
    # cuda = False so the generators don't use up all the memory
    imgs = dict_map(imgs, lambda v: torch_cuda(v, False))

    print('Loading generator ------------------------------------------------')
    generators = {}
    task_names = [e for e in os.listdir(paths['model']) if '2' in e]

    for i, nm in enumerate(task_names):
        path = os.path.join(paths['model'], nm, model_arch)
        ix = max([float(e.split('-')[1]) for e in os.listdir(path) if 'model_gen' in e])
        A_nm, B_nm = create_nms(nm, domain2lab)
        generators[A_nm] = torch.load(os.path.join(path, 'model_gen_A-' + str(ix)), map_location={'cuda:0': 'cpu'})
        generators[B_nm] = torch.load(os.path.join(path, 'model_gen_B-' + str(ix)), map_location={'cuda:0': 'cpu'})

    # translate all images (A and B)
    print('Translating images -----------------------------------------------')
    for ab in label_perms:
        print('{} to {}'.format(ab[0], ab[1]))
        imgs[ab] = generators[ab](imgs[ab[0]])

    # #########################################################################
    # TODO(lpupp) Is there around this? #######################################
    # The problem is the transform crap
    # Why do I need the transform stuff? because the encoder dimension expects
    # a different sized input.

    # Normalize all inputs to embedding_encoder
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

    imgs_enc = dict_map(imgs_enc, lambda v: minibatch_call(v, encoder))
    print(dict_map(imgs_enc, lambda v: v.shape))

    batch_size = 32
    all_imgs_enc = {}
    for k, v in all_img_paths.items():
        t_start = time.time()
        print('encoding all_imgs', k)
        out = []
        n_mb = math.ceil(len(v)/batch_size)
        for i in range(n_mb):
            dt = read_images(v[i * batch_size: (i+1) * batch_size], dsize[0])
            dt = torch_cuda(dt, cuda)
            out.append(encoder(dt))
        all_imgs_enc[k] = torch.cat(out, dim=0)
        print('time {}', time.time() - t_start)
    print(dict_map(all_imgs_enc, lambda v: v.shape))
    # TODO (lpupp) Could output this to csv...

    # #########################################################################

    print('Find top n similar -----------------------------------------------')
    print('Using discoGAN')
    # For each translation (AB and BA) find top n similarity (in B and A resp.)
    sim_disco, sim_vgg = {}, {}
    for ab in label_perms:
        sim_disco[ab] = find_top_n_similar(imgs_enc[ab], all_imgs_enc[ab[1]], n=topn)

    # TODO(lpupp) We don't need this comparison... do we?
    #print('Using pretrained {}'.format(args.embedding_encoder))
    #for a in domain_labs:
    #    for b in domain_labs:
    #        if a == b:
    #            tmp = find_top_n_similar(imgs_enc[a], all_imgs_enc[b], n=topn+1)
    #            sim_vgg[a+b] = dict_map(tmp, lambda v: v[:-1])
    #        else:
    #            sim_vgg[a+b] = find_top_n_similar(imgs_enc[a], all_imgs_enc[b], n=topn)

    # Plot results nicely
    print('Plotting results -------------------------------------------------')
    # Plot top n similar using discogan results
    for ab in label_perms:
        print(ab)
        a, b = ab[0], ab[1]
        plot_all_outputs(sim_disco[ab],
                         [imgs_np[a], imgs_np[ab], imgs_np[b]],
                         src_style=str(ab),
                         path=os.path.join(paths['topn'], domain[a], model_arch + '_in'))

    # Plot top n similar using VGG results
    #for ab in sim_vgg:
    #    print(ab)
    #    a, b = ab[0], ab[1]
    #    plot_all_outputs(sim_vgg[ab],
    #                     [imgs_np[a], np.ones_like(imgs_np[a]), imgs_np[b]],
    #                     src_style=str(ab),
    #                     path=os.path.join(paths['topn'], domain[a], 'vgg_in'))


def eval_random(img_size, topn, domain, paths):
    """Plot n random for discogan benchline."""
    d_nm = domain
    domain_set = domain_d[d_nm]
    domain_labs = letters[:len(domain_set)]

    domain = dict((k, v) for k, v in zip(domain_labs, domain_set))

    label_perms = [i + j for i, j in product(domain_labs, domain_labs) if i != j]

    for nm in domain.values():
        topn_path = os.path.join(paths['topn'], nm, 'random')
        if not os.path.exists(topn_path):
            os.makedirs(topn_path)

    img_paths = dict_map(domain, lambda v: get_photo_files(v)[1])
    all_img_paths = dict_map(domain, lambda v: train_val_photos(v))
    n_imgs = dict_map(img_paths, lambda v: len(v))
    n_all_imgs = dict_map(all_img_paths, lambda v: len(v))

    print('Plotting results -------------------------------------------------')
    random_ixs = {}
    for ab in label_perms:
        random_ixs.setdefault(ab, {})
        brange = list(range(n_all_imgs[ab[1]]))
        for k in range(n_imgs[ab[0]]):
            random_ixs[ab][k] = random.sample(brange, topn)

    for ab in random_ixs:
        print(ab)
        a, b = ab[0], ab[1]
        plot_all_random(random_ixs[ab], img_paths[a], all_img_paths[b],
                        img_size=img_size,
                        src_style=str(ab),
                        path=os.path.join(paths['topn'], domain[a], 'random'))


def eval_single_image(img_class, img_size, topn, encoder, cuda, model_arch, domain, paths, enc_img_size):
    """single_image TODO.

    Steps:
    - Load data.
    - Load generators.
    - Translate all images with generators.
    - Encode images with pretrained encoder.
    - Get top n.
    - Plot everything.
    """

    d_nm = domain
    domain_set = domain_d[d_nm]
    domain_set.insert(0, domain_set.pop(domain_set.index(img_class)))
    domain_labs = letters[:len(domain_set)]

    domain = dict((k, v) for k, v in zip(domain_labs, domain_set))
    domain2lab = dict((v, k) for k, v in domain.items())

    dsize = (enc_img_size, enc_img_size)

    topn_path = os.path.join(paths['topn'], 'single_img', model_arch)
    if not os.path.exists(topn_path):
        os.makedirs(topn_path)

    print('Reading images ---------------------------------------------------')
    source_img_np = read_image(paths['image'], img_size)
    source_img_np = np.expand_dims(source_img_np, 0)
    source_img = torch_cuda(source_img_np, cuda)

    img_paths = dict_map(domain, lambda v: get_photo_files(v)[1])
    img_paths['A'].remove(paths['image'])
    imgs_np = dict_map(img_paths, lambda v: read_images(v, img_size))

    print('Loading generator ------------------------------------------------')
    generators = {}
    task_names = [e for e in os.listdir(paths['model']) if img_class in e]

    for i, nm in enumerate(task_names):
        path = os.path.join(paths['model'], nm, model_arch)
        A, B = nm.split('2')
        ix = max([float(e.split('-')[1]) for e in os.listdir(path) if 'model_gen' in e])
        if A == img_class:
            nm = 'A' + domain2lab[B]
            generators[nm] = torch.load(os.path.join(path, 'model_gen_B-' + str(ix)))
        else:
            nm = 'A' + domain2lab[A]
            generators[nm] = torch.load(os.path.join(path, 'model_gen_A-' + str(ix)))

    print('Translating source image -----------------------------------------')
    img_trans = dict_map(generators, lambda gen: gen(source_img))

    print('Converting and resizing image tensors to numpy -------------------')
    img_trans_np = dict_map(img_trans, lambda v: as_np(v))

    img_src_enc = resize_array_of_images(source_img_np, dsize)
    img_trans_enc = dict_map(img_trans_np, lambda v: resize_array_of_images(v, dsize))
    imgs_enc = dict_map(imgs_np, lambda v: resize_array_of_images(v, dsize))

    img_src_enc = torch_cuda(img_src_enc, cuda)
    img_trans_enc = dict_map(img_trans_enc, lambda v: torch_cuda(v, cuda))
    imgs_enc = dict_map(imgs_enc, lambda v: torch_cuda(v, cuda))

    print('Encoding images --------------------------------------------------')
    img_src_enc = encoder(img_src_enc)
    img_trans_enc = dict_map(img_trans_enc, lambda v: encoder(v))
    imgs_enc = dict_map(imgs_enc, lambda v: minibatch_call(v, encoder))

    print('Find top n similar using discoGAN --------------------------------')
    sim = {'A': find_top_n_similar_by_img(torch.squeeze(img_src_enc), imgs_enc['A'], n=topn)}
    for k in img_trans_enc:
        sim[k[1]] = find_top_n_similar_by_img(torch.squeeze(img_trans_enc[k]), imgs_enc[k[1]], n=topn)

    sim_total = list((k, e) for k, v in sim.items() for e in v)
    topn_overall = sorted(sim_total, key=lambda kv: kv[1][1])[-topn:]

    print('Plotting results -------------------------------------------------')
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
