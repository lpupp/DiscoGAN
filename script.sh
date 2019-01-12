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
python ./discogan/image_translation.py --task_name=handbags2belts --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000
python ./discogan/image_translation.py --task_name=belts2shoes --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000
python ./discogan/image_translation.py --task_name=shoes2handbags --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000

# Run furniture
python ./discogan/image_translation.py --task_name=tables2seating --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000
python ./discogan/image_translation.py --task_name=seating2storage --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000
python ./discogan/image_translation.py --task_name=storage2tables --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000 --epoch_size 1000


# Evaluation -------------------------------------------
# Run fashion
#python ./discogan/inference_evaluation.py --task_name shoes2handbags --load_iter 18.0 --embedding_encoder vgg19
#python ./discogan/inference_evaluation.py --task_name handbags2belts --load_iter 18.0 --embedding_encoder vgg19
#python ./discogan/inference_evaluation.py --task_name belts2shoes --load_iter 18.0 --embedding_encoder vgg19

# Run furniture
#python ./discogan/inference_evaluation.py --task_name tables2chairs --load_iter 23.0 --embedding_encoder vgg19
#python ./discogan/inference_evaluation.py --task_name seating2storage --load_iter 23.0 --embedding_encoder vgg19
#python ./discogan/inference_evaluation.py --task_name storage2tables --load_iter 23.0 --embedding_encoder vgg19
