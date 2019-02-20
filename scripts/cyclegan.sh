#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres gpu:Tesla-K80:1

BASE_PATH=/home/cluster/lgaega/
GIT_PATH=${BASE_PATH}/perfectpairings/CycleGAN
DATA_PATH=${BASE_PATH}/data/CycleGAN/fashion/shoes2dresses
OUTPUT_DIR=${BASE_PATH}/output/perfectpairings/CycleGAN/shoes2dresses

RUN_NAME=iter1

source ~/tensorflow/bin/activate

# Training ---------------------------------------------
# Run fashion
srun python ${GIT_PATH}/train.py --dataroot ./datasets/maps --name shoes2dresses --model cycle_gan --checkpoints_dir ${OUTPUT_DIR}/checkpoints --no_html --no_flip >> ${OUTPUT_DIR}/${RUN_NAME}AtoB.txt
srun python ${GIT_PATH}/train.py --dataroot ./datasets/maps --name dresses2shoes --model cycle_gan --checkpoints_dir ${OUTPUT_DIR}/checkpoints --no_html --no_flip --direction BtoA >> ${OUTPUT_DIR}/${RUN_NAME}BtoA.txt

# Evaluation -------------------------------------------
# Run eval (out-of-sample)
#srun python ${GIT_PATH}/inference_evaluation.py --cuda

# Run eval (in-sample)
#srun python ${GIT_PATH}/inference_evaluation.py --eval_task in --cuda

# Run eval (random)
#srun python ${GIT_PATH}/inference_evaluation.py --eval_task random

# Merging output images --------------------------------
# Test merging
#python ${GIT_PATH}/summarize_imgs.py --task shoes
