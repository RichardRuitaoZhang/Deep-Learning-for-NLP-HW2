#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=hw2_generation
#SBATCH --output=logs/hw2_generation_%j.log

# ===== Working Directory =====
cd /projects/b1080/rz/cs461/HW2/

# ===== Environment Setup =====
module purge
conda activate transformer

# ===== Paths and Parameters =====
TRAIN_PATH=data/obqa.train.txt
VALID_PATH=data/obqa.valid.txt
SEQLEN=512
BATCH=2
LR=1e-4
EPOCHS=3
LOADNAME=pretrain

# ===== Run Fine-Tuning =====
echo "Starting GPT-based generative fine-tuning on GPU..."
nvidia-smi

python generation.py \
  --train_path $TRAIN_PATH \
  --valid_path $VALID_PATH \
  --seqlen $SEQLEN \
  --batch_size $BATCH \
  --lr $LR \
  --epochs $EPOCHS \
  --loadname $LOADNAME

echo "Generative fine-tuning completed. Model saved to models/generator.pt"
