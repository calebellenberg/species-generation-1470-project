#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 36:00:00
#SBATCH --mem=64000MB
#SBATCH --job-name='SpeciesGenerationCGAN'
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TF_FORCE_GPU_ALLOW_GROWTH=true
module purge
unset LD_LIBRARY_PATH
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"

CONTAINER_PATH="/oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg"
EXEC_PATH="srun apptainer exec --nv"

$EXEC_PATH $CONTAINER_PATH pip install --user --no-cache-dir pandas matplotlib

cd "${SLURM_SUBMIT_DIR}" || exit 1
echo "Running on node: $SLURM_NODELIST"
$EXEC_PATH $CONTAINER_PATH nvidia-smi

# $EXEC_PATH $CONTAINER_PATH python -u classifier.py
$EXEC_PATH $CONTAINER_PATH python -u binaryCGAN.py