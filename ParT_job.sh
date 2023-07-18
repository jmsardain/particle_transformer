#!/bin/bash --login
#$ -cwd              # Job will run in the current directory (where you ran qsub)
#$ -o ./logs
#$ -e ./logs
#$ -l nvidia_a100

echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"
export PYTHONUNBUFFERED=true

conda init bash
conda activate weaver
#export DATADIR_JetClass='/mnt/iusers01/fatpou01/phy01/j17668am/scratch/ConstituentTopTagging/Root2HDF5/dataloc'
export DATADIR_JetClass='/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer'
echo ${DATADIR_JetClass}

export DDP_NGPUS=$NGPUS
export COMMENT='v1'
export NUM_EPOCHS=1

# Next build command to run python training script
command="source train_JetClass.sh ParT kin"

# Run command
echo "================================"
echo "Will run command ${command}"
$command
echo -e "\nDone!"
