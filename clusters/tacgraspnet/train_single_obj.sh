#!/bin/bash
#SBATCH --partition=gpu              # Newton's GPU partition name
#SBATCH --time=12:00:00              # Runtime limit (HH:MM:SS)
#SBATCH --cpus-per-task=4            # Number of CPU cores for data loading
#SBATCH --mem=16G                    # Total RAM requested for the node
#SBATCH --gres=gpu:1                 # Request exactly 1 GPU
#SBATCH --mail-type=FAIL             # Send an email if the job fails
#SBATCH --mail-user=htruong@ec-lyon.fr
#SBATCH --output=logs/%x_%j.log   # Standard output log (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err    # Error log

# --- 1. Catch Command Line Arguments ---
# This catches the object name you pass when running 'sbatch'
OBJECT_NAME=$1

# If you forget to pass an argument, it will default to a placeholder
if [ -z "$OBJECT_NAME" ]; then
    echo "Error: No object name provided. Usage: sbatch train_objects.sh [object_name]"
    exit 1
fi

# --- 2. Create Logging Directory ---
# Ensures the directory for logs exists so Slurm doesn't crash
mkdir -p logs

# --- 3. Environment Setup (Newton Cluster Specific) ---
## Clean system modules to prevent conflicts
#module purge
#
## Load the modern Anaconda module available on Newton
#module load Anaconda3/2025.06-1

## Source the cluster-specific conda profile script to make 'conda activate' work inside Slurm
#CONDA_BASE=$(conda info --base)
#source "$CONDA_BASE/etc/profile.d/conda.sh"
#
## Activate your newly re-created internship environment
#conda activate m2internship
#
cd ~/m2internship/m2internship

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH}"

# --- 4. Execution ---
# Run your training file directly on the Newton filesystem
~/.conda/envs/m2internship/bin/python src/scripts/tacgraspnet/run.py
-m training
-ds single_obj
