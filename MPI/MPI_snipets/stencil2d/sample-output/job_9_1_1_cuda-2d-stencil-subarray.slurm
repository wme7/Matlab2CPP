#!/bin/bash
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2
#SBATCH --mem=24000
#SBATCH --time=00:30:00
#SBATCH --output=slurm_9_1_1_cuda-2d-stencil-subarray.out
. /etc/profile.d/modules.bash
module load mvapich2
module load cuda
module load gcc/4.6.3

mpiexec.hydra -rmk slurm ./cuda-2d-stencil-subarray     

