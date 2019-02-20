#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres gpu:Tesla-K80:1

BASE_PATH=/home/cluster/lgaega/
DATA_PATH=${BASE_PATH}/data

source ~/tensorflow/bin/activate

# UNZIP

srun python data_processing.py ${DATA_PATH}/fashion/shoes pad
srun python data_processing.py ${DATA_PATH}/fashion/dresses pad

srun python data_processing.py ${DATA_PATH}/fashion/shoes sample
srun python data_processing.py ${DATA_PATH}/fashion/dresses sample

# CycleGAN requires a different data structure.
cp -a ${DATA_PATH}/fashion/shoes/train ${DATA_PATH}/CycleGAN/fashion/shoes2dresses/trainA
cp -a ${DATA_PATH}/fashion/dresses/val ${DATA_PATH}/CycleGAN/fashion/shoes2dresses/testA
cp -a ${DATA_PATH}/fashion/shoes/train ${DATA_PATH}/CycleGAN/fashion/shoes2dresses/trainB
cp -a ${DATA_PATH}/fashion/dresses/val ${DATA_PATH}/CycleGAN/fashion/shoes2dresses/testB
