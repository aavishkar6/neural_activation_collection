#!/bin/bash

#================================================================#
#      SLURM DIRECTIVES                                          #
#================================================================#

#SBATCH --job-name=collect_activations   # A descriptive name for your job
#SBATCH --output=./activations_job_%j.out  # Standard output log file
#SBATCH --error=./activations_job_%j.err   # Standard error log file
#SBATCH --partition=nvidia           # The partition (queue) you want to use
#SBATCH --nodes=1                    # We need just one node
#SBATCH --ntasks-per-node=1          # One task (our python script) per node
#SBATCH --gres=gpu:a100:1            # Request one A100 GPU
#SBATCH -C 80g                       # Request 80 Gb explicitely
#SBATCH --mem=80GB                   # Request 80GB of system RAM
#SBATCH --time=96:00:00              # Maximum time for the job to run (96 hours)
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

# Parse command-line arguments
CONFIG_FILE="${1:-./config/base_config.yaml}"

# Run the collection script
python src/run_collection.py --config $CONFIG_FILE

# Print completion
echo ""
echo "End time: $(date)"
echo "Job completed"

