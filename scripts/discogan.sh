#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres gpu:Tesla-K80:1

BASE_PATH=/home/cluster/lgaega/
GIT_PATH=${BASE_PATH}/DiscoGAN/discogan
DATA_PATH=${BASE_PATH}/data/fashion
OUTPUT_DIR=${BASE_PATH}/output/perfectpairings/DiscoGAN/shoes2dresses

RUN_NAME=iter1

source ~/tensorflow/bin/activate

# Training ---------------------------------------------
# Run fashion
srun python ${GIT_PATH}/image_translation.py --task_name=shoes2dresses --starting_rate=0.5 --batch_size=256 --model_save_interval 1000 --epoch_size 1000 --cuda --result_path ${OUTPUT_DIR}/results --model_path ${OUTPUT_DIR}/models/ --plot_path ${OUTPUT_DIR}/plots/ >> ${OUTPUT_DIR}/${RUN_NAME}.txt

# Evaluation -------------------------------------------
# Run eval (out-of-sample)
srun python ${GIT_PATH}/inference_evaluation.py --cuda

# Run eval (in-sample)
srun python ${GIT_PATH}/inference_evaluation.py --eval_task in --cuda

# Run eval (random)
srun python ${GIT_PATH}/inference_evaluation.py --eval_task random

# Merging output images --------------------------------
# Test merging
python ${GIT_PATH}/summarize_imgs.py --task shoes
