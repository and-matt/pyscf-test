#!/bin/bash
#SBATCH --job-name=dft-lvc
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -euo pipefail

WORKDIR=${SLURM_JOB_ID}
mkdir -p runs/$WORKDIR
cd runs/$WORKDIR

module load Python
module load OpenBLAS
module load HDF5

JOBDIR=$(pwd)

python  $JOBDIR/source/dft-lvc.py \
        -ncpus $SLURM_CPUS_PER_TASK \
        -xyz $JOBDIR/structures/dimer_opt.xyz \
        > dft-lvc.out
