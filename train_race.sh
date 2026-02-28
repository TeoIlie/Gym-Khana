#!/bin/bash
#SBATCH --job-name=f1tenth_ppo
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=21:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=teodor.ilie@queensu.ca

module load python/3.11 scipy-stack gcc opencv
source ~/envs/f1tenth/bin/activate
cd ~/F1TENTH_Gym
python3 train/ppo_race.py --m t