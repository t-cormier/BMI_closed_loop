#!/bin/bash



#SBATCH -p IGIcuda1
#SBATCH --gres=gpu:GTX1080Ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
## time limit of 2 days
#SBATCH --time=3-00:00:00
## Slurm emails you when the job starts/finishes/fails
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<titouan.slurm@gmail.com>
## %J is the job number
#SBATCH --output=sbatch_out/out_per_trial_sim-%J.out
#SBATCH --error=sbatch_out/err_per_trial_sim-%J.err

source ~/anaconda/bin/activate root
conda activate tf_slurm
srun --ntasks=1 python main.py
