#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=cur_roll304
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/Odyssey/private/d21botvy/oceanbench_eval/logs_currents/%x_%j.log
# Geostrophic + cyclogeostrophic (jaxparrow) currents of the 8 SWOT-paper forecasts at the hourly drifter points,
# then the analysis tables. Usage: sbatch run_match.sh
export HOME=/Odyssey/private/d21botvy/
PY=/Odyssey/private/d21botvy/venvs/cyclo/bin/python
S=/Odyssey/private/d21botvy/swot-paper-repro/currents/drifter_cyclo_eval.py
cd /Odyssey/private/d21botvy/oceanbench_eval
$PY -u $S match fdvroll304_nadir fdvroll304_nadir_swot; echo "match EXIT $?"
