# Run a specific task by uncommenting corresponding line

#conda create -n py2 python=2.7 numpy progressbar2 scipy pandas
#conda install -n py2 -c pytorch pytorch
#conda install -n py2 -c menpo opencv
#conda install -n py2 -c anaconda pil

#activate py2

#python data_processing.py /Users/lucagaegauf/Documents/GitHub/Keras-GAN/discogan/datasets/fashion/belts pad
#python data_processing.py /Users/lucagaegauf/Documents/GitHub/Keras-GAN/discogan/datasets/fashion/belts sample

# TODO(lpupp) I first need to select only those with white background
#python data_processing.py /Users/lucagaegauf/Documents/GitHub/Keras-GAN/discogan/datasets/furniture/storage sample

# Training ---------------------------------------------
# Run fashion
#python ./discogan/image_translation.py --task_name=handbags2belts --starting_rate=0.5 --batch_size=256 --model_save_interval 1000 --epoch_size 1000 --cuda
#python ./discogan/image_translation.py --task_name=belts2shoes --starting_rate=0.5 --batch_size=256 --model_save_interval 1000 --epoch_size 1000 --cuda
#python ./discogan/image_translation.py --task_name=shoes2handbags --starting_rate=0.5 --batch_size=256 --model_save_interval 1000 --epoch_size 1000 --cuda
#python ./discogan/image_translation.py --task_name=dresses2belts --starting_rate=0.5 --batch_size=256 --model_save_interval 1000 --epoch_size 1000 --cuda
#python ./discogan/image_translation.py --task_name=dresses2handbags --starting_rate=0.5 --batch_size=256 --model_save_interval 1000 --epoch_size 1000 --cuda
#python ./discogan/image_translation.py --task_name=shoes2dresses --starting_rate=0.5 --batch_size=256 --model_save_interval 1000 --epoch_size 1000 --cuda

# Run furniture
#python ./discogan/image_translation.py --task_name=tables2seating --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000 --cuda
#python ./discogan/image_translation.py --task_name=seating2storage --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000 --cuda
#python ./discogan/image_translation.py --task_name=storage2tables --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000 --cuda

# Evaluation -------------------------------------------
# Run eval (out-of-sample)
#python discogan/inference_evaluation.py --cuda
#python discogan/inference_evaluation.py --domain furniture --cuda

# Run eval (in-sample)
python discogan/inference_evaluation.py --eval_task in --cuda
python discogan/inference_evaluation.py --domain furniture --eval_task in --cuda

# Run eval (single image)
#python discogan/inference_evaluation.py --image_path C:\Users\lucag\Dropbox\GAN\fashion\shoes\val\2_AB.jpg --image_class shoes
#python discogan/inference_evaluation.py --domain furniture --image_path C:\Users\lucag\Dropbox\GAN\furniture\tables\convito_5_1_sq.jpg --image_class tables

# Run eval (random)
#python discogan/inference_evaluation.py --eval_task random
#python discogan/inference_evaluation.py --domain furniture --eval_task random

# Merging output images --------------------------------
# Test merging
python discogan/summarize_imgs.py --task shoes
