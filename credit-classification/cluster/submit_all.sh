#!/bin/bash
# Submits the complete Section 5.3 experiment (5 values of N, 50 simulations, 6 methods) from the fusion-DRO
# directory. For each N, one job computes and certifies the releases, and the training jobs start once it succeeds.
# The grouping of simulations per job matches the runs reported in the paper (e.g. ten simulations per AFFINE job).
set -euo pipefail
JOB=credit-classification/cluster/job.sbatch
mkdir -p credit-classification/results/logs
for N in 2000 6000 10000 14000 18000; do
  rel=$(sbatch --parsable --job-name="rel-$N" --time=01:00:00 --export=ALL,N=$N,SIMS=0-49,RELEASES=1 $JOB)
  after="--dependency=afterok:$rel"
  for M in affine static; do for S in 0-9 10-19 20-29 30-39 40-49; do
    sbatch $after --job-name="$M-$N" --time=03:00:00 --export=ALL,N=$N,METHOD=$M,SIMS=$S $JOB; done; done
  for M in minkowski tv; do for S in 0-24 25-49; do
    sbatch $after --job-name="$M-$N" --time=03:00:00 --export=ALL,N=$N,METHOD=$M,SIMS=$S $JOB; done; done
  sbatch $after --job-name="moment-$N" --time=01:30:00 --export=ALL,N=$N,METHOD=moment,SIMS=0-49 $JOB
  sbatch $after --job-name="oracle-$N" --time=01:00:00 --export=ALL,N=$N,METHOD=oracle,SIMS=0-49 $JOB
done
