#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres gpu:Tesla-K80:1

BASE_PATH=/home/cluster/lgaega/
GIT_PATH=${BASE_PATH}/perfectpairings/StarGAN
EVAL_PATH=${BASE_PATH}/DiscoGAN/discogan
DATA_PATH=${BASE_PATH}/data/fashion
OUTPUT_DIR=${BASE_PATH}/output/perfectpairings/StarGAN/shoes2dresses

RUN_NAME=iter1

source ~/tensorflow/bin/activate

# Training ---------------------------------------------
# Run fashion
srun python ${GIT_PATH}/image_translation.py --realreal_image_dir ${DATA_PATH} --use_tensorboard False --num_workers 2 --log_dir ${OUTPUT_DIR}/logs --model_save_dir ${OUTPUT_DIR}/models --sample_dir ${OUTPUT_DIR}/samples --result_dir ${OUTPUT_DIR}/results >> ${OUTPUT_DIR}/${RUN_NAME}.txt

# This has to be to new directory
# Evaluation -------------------------------------------
# Run eval (out-of-sample)
#srun python ${EVAL_PATH}/inference_evaluation.py --cuda

# Run eval (in-sample)
#srun python ${EVAL_PATH}/inference_evaluation.py --eval_task in --cuda

# Run eval (random)
#srun python ${EVAL_PATH}/inference_evaluation.py --eval_task random

# Merging output images --------------------------------
# Test merging
#python ${EVAL_PATH}/summarize_imgs.py --task shoes
