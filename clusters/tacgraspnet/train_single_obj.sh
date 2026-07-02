#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hoang-nguyen-vu.truong@ec-lyon.fr
#SBATCH --output=../logs/%x_%j.log
#SBATCH --error=../logs/%x_%j.err


OBJECT_NAME=$1

if [ -z "$OBJECT_NAME" ]; then
    echo "Error: No object name provided. Usage: sbatch train_single_obj.sh [object_name]"
    exit 1
fi

mkdir -p ~/m2internship/m2internship/clusters/logs

module purge
module load CUDA/12.8.0

cd ~/claude_m2internship/m2internship

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH}"


~/.conda/envs/m2internship/bin/python scripts/tacgraspnet/run.py \
-m training \
-ds single_obj \
-o "$OBJECT_NAME" \
# Validation ratio
-vs 0.2 \
# Batch size
-bs 1 \
# Number of epochs
-ne 20 \
# Learning rate
-lr 8e-5 \
# Use template data
-td False \
# Use final layer norms
-fln False \
# Normalize features
-nf True \
# Normalize outputs
-no True \
# Use separate node and tetrahedral decoders
-ntsd True \
# Use separate edge MLPs
-sem False \
# Use separate message passing MLPs (GraphNetBlock)
-mpsm False \
# Use global node
-gn False \
# Use translation inductive bias
-tib False \
# Message passing steps
-mps 15 \
# Radius to construct contact edges
-r 0.005 \
