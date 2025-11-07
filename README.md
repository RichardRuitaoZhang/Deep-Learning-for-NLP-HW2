# CS461: Deep Learning for NLP — HW2

This repository contains the full implementation and environment setup for Homework 2 of CS461: Deep Learning for Natural Language Processing (Fall 2025).  
All scripts are designed for GPU execution and follow the standard structure for classification (Part 1) and generation (Part 2) tasks.

---

## Directory Structure

```
HW2/
├── __pycache__/                # Python cache files
├── data/                       # Input datasets for training and validation
│   ├── obqa.train.txt
│   ├── obqa.valid.txt
│   ├── obqa.test.txt
│   ├── wiki.train.txt
│   ├── wiki.valid.txt
│   └── wiki.test.txt
├── logs/                       # Training and evaluation log outputs
├── models/                     # Saved checkpoints and model weights
│   ├── classifier.pt
│   ├── generator.pt
│   └── model_weights
├── pretrain/                   # Pretrained transformer models (downloaded or cached)
│   └── model_weights
├── scripts/                    # SLURM/GPU training shell scripts
│   ├── train_classification_gpu.sh
│   └── train_generation_gpu.sh
├── utils/                      # Reusable utility modules
│   ├── metrics.py              # Accuracy/metrics helpers
│   └── utils_data.py           # Dataset loading & preprocessing
├── .gitattributes              # Git LFS configuration
├── README.md                   # Project documentation
├── classification.py           # Classification task pipeline
├── generation.py               # Generative model pipeline
├── starter.py                  # Transformer blocks & shared components
└── transformer.yml             # Conda environment specification
```
---

## Environment Setup

**1. Create the conda environment from the provided YAML file:**

       conda env create -f transformer.yml
       conda activate transformer

**2. Ensure the following key packages are installed (already included in the transformer.yml):**

| Package | Version | Description |
|----------|----------|-------------|
| **torch** | 2.8.0 | Core PyTorch deep learning framework (CUDA 12 enabled) |
| **transformers** | 4.45.2 | Hugging Face Transformers for BERT/GPT models |
| **datasets** | 3.0.2 | Dataset loader and preprocessing tools |
| **evaluate** | 0.4.2 | Evaluation metrics for NLP tasks |
| **accelerate** | 1.1.1 | Multi-GPU / distributed training support |
| **numpy** | 2.2.6 | Numerical computation backend |
| **pandas** | 2.2.x *(implied)* | Data manipulation and analysis |
| **scikit-learn** | 1.5.x *(optional)* | Metrics and preprocessing utilities |
| **huggingface-hub** | 0.26.2 | Sync models/datasets with the Hugging Face Hub |
| **tokenizers** | 0.20.0 | Fast tokenization backend for Transformers |


**3. CUDA must be available and compatible with your PyTorch build for GPU training.**

---

## Training Scripts

All GPU-based training is handled through shell scripts under `scripts/`.

### Classification Task (Part 1)
Run:

       sbatch scripts/train_classification_gpu.sh

- Loads dataset from `data/`, which contains `obqa.train.txt` `obqa.valid.txt` `obqa.test.txt` `wiki.train.txt` `wiki.valid.txt` `wiki.test.txt`
- Initializes model defined in `classification.py`  
- Logs training progress to `logs/`  
- Saves checkpoints to `models/`

### Generative Task (Part 2)
Run:

       sbatch scripts/train_generation_gpu.sh

- Executes generative model training as implemented in `generation.py`  
- Uses `starter.py` from David (moved to the root) for core Transformer definitions  
- Outputs logs and checkpoints to their respective directories  

---

## Logs

All training and validation metrics are automatically stored in the `logs/` directory.  
Each log file includes a timestamp and task type for reproducibility.

---

## Models

The `models/` directory stores:
- Intermediate checkpoints (`.pt` or `.bin`)  
- Final trained model weights  
- Any exported Hugging Face–compatible model files for submission  

Ensure to compress this directory for final submission if model weights are required.

---

## Pretrained Models

The `pretrain/` directory contains cached pretrained backbones `model_weights` from David's 'HW#2'.  
If not present, they will be automatically downloaded via the Hugging Face hub during the first run.

---

## Notes

- All paths are relative to the root `HW2/` directory.  
- Both training scripts are GPU-optimized and compatible with the provided conda environment.  
- Logs and model outputs are automatically organized; manual cleanup is optional.  
- No additional external datasets or fine-tuned checkpoints are required beyond what’s provided.

---

**Author:** Ruitao Zhang  
**Course:** CS461 — Deep Learning for NLP, Fall 2025  
**Instructor:** Prof. David Demeter
