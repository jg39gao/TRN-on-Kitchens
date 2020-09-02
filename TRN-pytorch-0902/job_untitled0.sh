#!/bin/bash
#$ -l h_rt=100:00:00  #time needed
#$ -pe smp 1 #number of cores
#$ -l rmem=16G #number of memery per cpu cores.
#$ -l gpu=1  #number of GPUs per CPU core
#$ -o ./log/test1_output.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M jgao39@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory


module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
module load libs/CUDA/10.2.89/binary

source activate jupyter-spark


python untitled0.py 