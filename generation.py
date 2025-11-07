import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from starter import get_modelGPT
from utils.metrics import evaluate_generator  # external evaluation


# Build training data for GPT fine-tuning
# Each instance -> "[START] fact question [A] ... [B] ... [C] ... [D] ... [ANSWER]"
class OpenBookQAGenDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=512, train=True):
        self.samples = []
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.labels = ["A", "B", "C", "D"]
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("|") if p.strip()]
                if len(parts) != 7:
                    continue
                fact, question, A, B, C, D, ans = parts
                label = label_map.get(ans, -1)
                if label == -1:
                    continue
                if train:
                    text = (
                        f"[START] {fact} {question} "
                        f"[A] {A} [B] {B} [C] {C} [D] {D} [ANSWER] {self.labels[label]}"
                    )
                else:
                    text = (
                        f"[START] {fact} {question} "
                        f"[A] {A} [B] {B} [C] {C} [D] {D} [ANSWER]"
                    )
                ids = tokenizer.encode(text, truncation=True, max_length=seq_len)
                self.samples.append((torch.tensor(ids), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# simple padding collate
def collate_fn(batch):
    ids, labels = zip(*batch)
    max_len = max(len(x) for x in ids)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(ids):
        padded[i, : len(x)] = x
    return padded, torch.tensor(labels, dtype=torch.long)


# standard fine-tuning loop (loss only on last token)
def train(model, dataloader, criterion, optimizer, device, id_list):
    model.train()
    total_loss = 0
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(input_ids, None)
        last_logits = logits[:, -1, :]     # predict next token after [ANSWER]
        target_ids = id_list[labels]       # A/B/C/D tokens
        loss = criterion(last_logits, target_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/obqa.train.txt")
    parser.add_argument("--valid_path", default="data/obqa.valid.txt")
    parser.add_argument("--loadname", default="pretrain")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", type=float, default=2.0)
    parser.add_argument("--tied", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    opt = args
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"Using device: {torch.cuda.get_device_name(0)}") # print gpu info

    vocab_size = len(tokenizer)
    args.vocab_size = vocab_size
    args.indices = torch.arange(vocab_size).to(opt.device)

    model = get_modelGPT(args, vocab_size).to(opt.device)

    train_set = OpenBookQAGenDataset(args.train_path, tokenizer, args.seqlen, train=True)
    val_set = OpenBookQAGenDataset(args.valid_path, tokenizer, args.seqlen, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_fn)

    # map letters to vocab token ids
    ans_ids = {k: tokenizer.encode(" " + k, add_special_tokens=False)[-1] for k in "ABCD"}
    id_list = torch.tensor([ans_ids[c] for c in "ABCD"], device=opt.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train(model, train_loader, criterion, optimizer, opt.device, id_list)
        acc = evaluate_generator(model, val_loader, id_list, opt.device)
        print(f"Epoch {epoch+1}: train_loss={loss:.4f}, val_acc={acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/generator.pt")
    print("Model saved to models/generator.pt")
    # running time
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
