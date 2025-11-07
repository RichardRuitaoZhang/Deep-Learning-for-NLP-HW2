#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=hw2_classification
#SBATCH --output=logs/hw2_classification_%j.log

# ===== Working Directory =====
cd /projects/b1080/rz/cs461/HW2/

# ===== Environment Setup =====
module purge
conda activate transformer

# ===== Paths and Parameters =====
TRAIN_PATH=data/obqa.train.txt
VALID_PATH=data/obqa.valid.txt

# ===== Run Training =====
echo "Starting classification fine-tuning on GPU..."
nvidia-smi

python classification.py \
  --train_path $TRAIN_PATH \
  --valid_path $VALID_PATH \
  --batch_size 8 \
  --epochs 3 \
  --lr 2e-5 \
  --max_length 256

echo "Classification training completed. Model saved to models/classifier.pt"
