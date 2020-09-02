#!/bin/bash                                                                
#$ -l gpu=2
#$ -l rmem=20G # 16g batchsize  4
#$ -o ./log/base_RGB.output 
#$ -l h_rt=95:00:00
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M jgao39@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
module load libs/CUDA/10.2.89/binary
source activate jupyter-spark



CUDA_VISIBLE_DEVICES=0,1 python main.py  epic RGB \
                     --epochs 10 --arch BNInception --num_segments 4  --print_model True \
                     --consensus_type TRN --batch_size 12  --workers 2 \
                     --lr 0.0001 \
                     --no_partialbn False 