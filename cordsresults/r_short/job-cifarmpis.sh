#!/bin/bash
#COBALT -q full-node
#COBALT -n 1
#COBALT -t 120
#COBALT -A datascience
#COBALT --attrs filesystems=home,grand,eagle,theta-fs0
# Necessary for Bash shells
. /etc/profile

# Tensorflow optimized for A100 with CUDA 11
module load conda/pytorch
# module load conda/pytorch

# Activate conda env
conda activate pycords
# conda activate base
export PYTHONPATH=/lus/grand/projects/datascience/ianwixom/expcifar:$PYTHONPATH

# User Configuration
# INIT_SCRIPT=$PWD/activate-dh.sh
COBALT_JOBSIZE=1
RANKS_PER_NODE=8

# Initialization of environment
# source $INIT_SCRIPT
module list
mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python cifar10cordsmodel.py