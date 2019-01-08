# Run a specific task by uncommenting corresponding line

conda create -n py2 python=2.7 numpy progressbar2 scipy pandas
conda install -n py2 -c pytorch pytorch
conda install -n py2 -c menpo opencv
conda install -n py2 -c anaconda pil

activate py2

# Run Shoes2Handbags
python ./discogan/image_translation.py --task_name=shoes2handbags --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000

# Run Handbags2Shoes
# python ./discogan/image_translation.py --task_name='handbags2shoes' --batch_size=500

# Run Tables2Chairs
python ./discogan/image_translation.py --task_name=tables2chairs --starting_rate=0.5 --batch_size=256 --cuda=true --model_save_interval 1000

# Run eval
./discogan/inference_evaluation.py --task_name shoes2handbags --load_iter 18.0 --embedding_encoder vgg19

./discogan/inference_evaluation.py --task_name tables2chairs --load_iter 23.0 --embedding_encoder vgg19
