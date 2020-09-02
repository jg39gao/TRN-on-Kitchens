#!/bin/bash                                                                
#$ -l gpu=1
#$ -l rmem=16G
#$ -o ./log/untitled_output.output 
#$ -l h_rt=96:00:00
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M jgao39@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

#cd /fastdata/acq18jg/epic/Code/TRN-pytorch-master/

module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
module load libs/CUDA/10.2.89/binary
source activate jupyter-spark
python untitled0.py 