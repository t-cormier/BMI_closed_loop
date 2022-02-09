#!/bin/bash
#SBATCH --mail-user=titouan.slurm@gmail.com
#SBATCH --mail-type=NONE
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name="bash"
#SBATCH --partition=IGIpcluster
#SBATCH --output=sbatch_out/out_jupyter_lab.out
#SBATCH --error=sbatch_out/err_jupyter_lab.err

NB_PORT=5050
HOST=$(hostname -s)
source ~/anaconda/bin/activate root
conda activate tensorflow_2.6.0
python -m jupyter lab --no-browser --port $NB_PORT
