#!/bin/bash -x
#SBATCH --account=covidnetx
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
source set_torch_distributed_vars.sh
source scripts/init_2020.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Job id: $SLURM_JOB_ID"
srun python -u $*