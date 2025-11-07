import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class OpenBookQADataset(Dataset):
    """
    Custom Dataset for OpenBookQA in original .txt format.
    -----------------------------------------------------
    Each line in obqa.train.txt / obqa.valid.txt / obqa.test.txt looks like:
        fact<TAB>question<TAB>choiceA<TAB>choiceB<TAB>choiceC<TAB>choiceD<TAB>answer

    Example:
        Plants need sunlight.   What do plants need?   Water   Sunlight   Food   Heat   B

    This Dataset:
        1. Reads the tab-separated text lines;
        2. Tokenizes each (fact + question + choice) pair using a BERT tokenizer;
        3. Returns tensors of shape:
            input_ids:      (num_choices, seq_len)
            attention_mask: (num_choices, seq_len)
            labels:         scalar index (0–3)
    """

    def __init__(self, path, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 7:
                    # Skip malformed lines
                    continue

                fact, question, A, B, C, D, ans = parts
                choices = [A, B, C, D]
                label = label_map.get(ans, -1)
                if label == -1:
                    continue

                # Tokenize each choice: [CLS] fact question choice [SEP]
                encodings = []
                for choice in choices:
                    text = f"{fact} {question} {choice}"
                    enc = tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    encodings.append(enc)

                self.samples.append((encodings, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encodings, label = self.samples[idx]
        input_ids = torch.stack([e["input_ids"].squeeze(0) for e in encodings])
        attention_mask = torch.stack([e["attention_mask"].squeeze(0) for e in encodings])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_tokenizer(model_name="bert-base-uncased"):
    """
    Utility wrapper to initialize tokenizer from model name.
    Keeps consistency across different scripts.
    """
    return AutoTokenizer.from_pretrained(model_name)
