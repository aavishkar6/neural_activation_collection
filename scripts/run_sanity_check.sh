#!/bin/bash

#================================================================#
#      SLURM DIRECTIVES                                          #
#================================================================#

#SBATCH --job-name=collect_activations   # A descriptive name for your job
#SBATCH --output=./logs/activations_job_%j.out  # Standard output log file
#SBATCH --error=./logs/activations_job_%j.err   # Standard error log file
#SBATCH --partition=nvidia           # The partition (queue) you want to use
#SBATCH --nodes=1                    # We need just one node
#SBATCH --ntasks-per-node=1          # One task (our python script) per node
#SBATCH --gres=gpu:a100:2            # Request one A100 GPU
#SBATCH -C 40g                       # Request 80 Gb explicitly
#SBATCH --mem=40GB                   # Request 80GB of system RAM
#SBATCH --time=04:00:00              # Maximum time for the job to run (96 hours)
#================================================================#
#      ENVIRONMENT SETUP                                         #
#================================================================#


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

# Load modules
module purge
module load miniconda

# Initialize the shell for Conda
eval "$(conda shell.bash hook)"

# Activate Conda environment
conda activate abliteration

# Run the collection script
python test_gemma_27b.py

# Print completion
echo ""
echo "End time: $(date)"
echo "Job completed"