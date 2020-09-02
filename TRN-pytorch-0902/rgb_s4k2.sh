#!/bin/bash                                                                
#$ -l gpu=2
#$ -l rmem=20G
#$ -o ./log/RGB_S4_k2_epoch40.output
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
                     --epochs 40 --arch BNInception --num_segments 4 --new_length 2 --print_model True \
                     --consensus_type TRN --batch_size 6  --workers 6 \
                     --resume model/TRN_epic_RGB_BNInception_TRN_segment4_K2_best.pth.tar \
                     --lr 0.0001 \
                     --no_partialbn False 