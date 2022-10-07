#!/bin/bash
#SBATCH --job-name=zero_combo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=13:00:00
#SBATCH --account=def-ssanner
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-user=<armin.toroghi@mail.utoronto.ca>
#SBATCH --mail-type=ALL

#wandb offline
#wandb online -num_run 
#wandb agent atoroghi/pre-critiquing/xydr8uhv
#python3 launch.py -epochs 1 -save_each 1 -kg no_kg
source ~/projects/def-ssanner/atoroghi/project/ENV/bin/activate
cd ~/projects/def-ssanner/atoroghi/project/BK-KGE
#python3 nested_cv_zero.py -tune_name soft_best -reg_type gauss -loss_type softplus -reduce_type sum -optim_type adagrad -sample_type split_rev -init_type uniform -kg kg 
python3 nested_cv_zero.py -tune_name soft_reg -reg_type gauss -loss_type softplus -reduce_type sum -optim_type adagrad -sample_type split_reg -init_type uniform -kg kg 