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
-vs 0.2 \ # Validation ratio
-bs 1 \ # Batch size
-ne 20 \ # Number of epochs
-lr 8e-5 \ # Learning rate
-td False \ # Use template data
-fln False \ # Use final layer norms
-nf True \ # Normalize features
-no True \ # Normalize outputs
-ntsd True \ # Use separate node and tetrahedral decoders
-sem False \ # Use separate edge MLPs
-mpsm False \ # Use separate message passing MLPs (GraphNetBlock)
-gn False \ # Use global node
-tib False \ # Use translation inductive bias
-mps 15 \ # Message passing steps
-r 0.005 \ # Radius to construct contact edges
