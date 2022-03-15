#!/bin/bash

#SBATCH --job-name=proxy_distribution
#SBATCH --output=slurm_logs/job-%A_%a.out
#SBATCH --error=slurm_logs/job-%A_%a.err

#SBATCH --nodes=8                # node count (number of different machine)
#SBATCH --ntasks-per-node=2      # number of tasks per-node (choose equal to gpus) [make sure ntasks and ngpus are equal]
#SBATCH --gpus-per-node=2        # gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)

export WORLD_SIZE=16 # set it equal to total number of gpus across all nodes (=total number of tasks across all nodes)

# Borrowed from https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=8120
echo "NODELIST="${SLURM_JOB_NODELIST}
echo "MASTER_ADDR="$MASTER_ADDR
echo $(date '+%d/%m/%Y %H:%M:%S')

# move data to local /scratch for faster loading. REPLACE it with local-scratch on your cluster. 
# I needed it because loading millions of small files from /data on our clusters was too slow. 
# NOTE that there is no need to bring data to local scatach if the loading speed is already good enough.
# In that case, please delete these next eight lines. 
for node in $(scontrol show hostnames $SLURM_JOB_NODELIST)
do
    ssh $node
    mkdir /scratch/serialized/
    echo "moving data to scratch on node ${node}"
    cp -r /scratch/gpfs/vvikash/synthetic_robustness/synthetic_dataset/diffusion/serialized/* /scratch/serialized/
    echo "moved data to scratch on node ${node}"
done


module load anaconda3/2020.11
source activate py37 # use your conda environment

# Batch-size will be splitted over all gpus. To keep batch-norm stable, use a large batch-size (atleast 16/gpu). 
# Otherwise enable --sync-bn

echo $CUDA_VISIBLE_DEVICES
srun python train.py --dataset cifar10 --arch resnest152 --data-dir /tigress/vvikash/datasets/all_cifar10/cifar10_pytorch/ \
    --trainer pgd --val-method pgd --batch-size 256 --batch-size-syn 256 --lr 0.2 --syn-data-list ddpm_cifar10 \
    --syn-data-dir /scratch/serialized/ --exp-name cifar10_resnest152_pgd_advtrain_ddpm_cifar10_serialized