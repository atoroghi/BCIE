
#!/bin/bash
#SBATCH --time=0:00:02
#SBATCH --account=def-ssanner
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=<armin.toroghi@mail.utoronto.ca>
#SBATCH --output=%x-%j.out

source /home/atoroghi/projects/def-ssanner/atoroghi/project/ENV/bin/activate
cd /home/atoroghi/projects/def-ssanner/atoroghi/project/BK-KGE-Mine

python main-fake.py