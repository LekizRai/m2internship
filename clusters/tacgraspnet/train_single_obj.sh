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

cd ~/m2internship/m2internship

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH}"


~/.conda/envs/m2internship/bin/python scripts/tacgraspnet/run.py \
-m training \
-ds single_obj
