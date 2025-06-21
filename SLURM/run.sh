#!/bin/bash
#SBATCH --job-name=team_name
#SBATCH --partition=hackathon
#SBATCH -A hackathon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs_team_name/%x-%j.out

source ~/miniconda3/bin/activate <your_env_name>
python3 <your_script_name>.py \
    -- flags
