#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p jlessler
#SBATCH --mem=64G
#SBATCH -t 00-04:00:00
#SBATCH --array=1,2,3
#SBATCH --gres=gpu:1

# 1 node (N) 1 task (n) to the general partition  for 7days, with this task needing 6 cpus
module purge
#./$CONDA_ROOT/etc/profile.d/conda.sh
#module add anaconda
#conda init
#conda activate pymc4_env > test.out

/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python -u main_refine.py --spec_id ${SLURM_ARRAY_TASK_ID} --inpaint True > out_refine_${SLURM_ARRAY_TASK_ID}.out 2>&1


