DiscoGAN
=========================================

Python 3.6 compatible DiscoGAN. Currently all applications other than handbags2shoes have been removed.

Source paper: [Learning to Discover Cross-Domain Relations
with Generative Adversarial Networks](https://arxiv.org/pdf/1703.05192.pdf).

Source git: [DiscoGAN](https://github.com/SKTBrain/DiscoGAN).

Table of Contents
-------------
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Training DiscoGAN](#trainingdiscogan)
    + [Handbags / Shoes](#h2s)
    + [Other](#other)
  * [Outputs](#outputs)

Prerequisites
-------------
   - Python 3.6
   - PyTorch
   - Numpy/Scipy/Pandas
   - Progressbar
   - OpenCV
   
Installation
-------------
    $ git clone https://github.com/lpupp/DiscoGAN
    $ cd DiscoGAN/
    $ sudo pip3 install -r requirements.txt

Training DiscoGAN
----------------
### Handbags / Shoes
Download edges2handbags dataset using 

    $ python ./datasets/download.py edges2handbags

Download edges2shoes dataset using 

    $ python ./datasets/download.py edges2shoes

Since we are not interested in the Edges we can remove them from the images completely to save space using

    $ python ./discogan/extract_imgs.py edges2handbags
    $ python ./discogan/extract_imgs.py edges2shoes

To train handbags2shoes or shoes2handbags (identical), set task_name

    $ python ./discogan/image_translation.py --task_name='shoes2handbags' --starting_rate=0.5

### Other
TBC...

Outputs
=============
All example results show x_A, x_AB, x_ABA and x_B, x_BA, x_BAB
