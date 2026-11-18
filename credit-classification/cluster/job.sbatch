#!/bin/bash
# One Slurm job of the Section 5.3 experiment: one N and one method for a range of simulations, all trained in
# parallel on one GPU. With RELEASES=1 the job only computes and certifies the releases of the simulations.
#   sbatch --export=ALL,N=6000,METHOD=affine,SIMS=0-9 credit-classification/cluster/job.sbatch
# Submit from the fusion-DRO directory with the Python environment of credit-classification/requirements.txt active.
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=credit-classification/results/logs/%x_%j.out
if [ "${RELEASES:-0}" = 1 ]; then
  python -u credit-classification/run_classification.py --n "$N" --sims "$SIMS" --releases_only
else
  python -u credit-classification/run_classification.py --n "$N" --sims "$SIMS" --methods "$METHOD"
fi
