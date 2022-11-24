#!/bin/bash
#SBATCH --job-name=zero
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --account=rrg-ssanner
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=<armin.toroghi@mail.utoronto.ca>
#SBATCH --mail-type=ALL

#wandb offline
#wandb online -num_run 
#wandb agent atoroghi/pre-critiquing/xydr8uhv
#python3 launch.py -epochs 1 -save_each 1 -kg no_kg
source ~/projects/def-ssanner/atoroghi/project/ENV/bin/activate
cd ~/projects/def-ssanner/atoroghi/project/BK-KGE
#python3 nested_cv_zero.py -tune_name soft_best -reg_type gauss -loss_type softplus -reduce_type sum -optim_type adagrad -sample_type split_rev -init_type uniform -kg kg 
#python3 nested_cv_zero.py -tune_name svd_neg_soft -reg_type gauss -loss_type softplus -reduce_type sum -optim_type adagrad -sample_type split_reg -init_type uniform -learning_rel freeze -kg no_kg 
python3 inner_cv.py -tune_name gaussindirect -fold 0
