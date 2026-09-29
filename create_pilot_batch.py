from pathlib import Path

content = """#!/bin/bash
#SBATCH --job-name=se801_loader
#SBATCH --partition=argon_ai
#SBATCH --time=00:08:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/szain/SE801_FMCW/loader_%j.log

set -e
source /etc/profile.d/lmod.sh
module use /opt/atlas/modulefiles/
module load 3.12/pt_base

python -u /home/szain/SE801_FMCW/pilot_fold_loader.py --outer 1 --inner 1
"""

Path(__file__).with_name("pilot_fold_loader.sbatch").write_bytes(
    content.encode("utf-8")
)
print("Created pilot_fold_loader.sbatch with Unix line endings")