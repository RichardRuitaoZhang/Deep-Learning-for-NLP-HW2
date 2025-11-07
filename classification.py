import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from utils.metrics import classifier_accuracy  # evaluation is handled externally


# Dataset for multiple-choice QA
# Each sample: fact | question | choiceA | choiceB | choiceC | choiceD | answer
class OpenBookQADataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("|") if p.strip()]
                if len(parts) != 7:
                    continue
                fact, question, A, B, C, D, ans = parts
                label = label_map.get(ans, -1)
                if label == -1:
                    continue

                # encode each (fact, question, choice) pair
                encodings = []
                for choice in [A, B, C, D]:
                    text = f"{fact} {question} {choice}"
                    enc = tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt",
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
            "labels": torch.tensor(label, dtype=torch.long),
        }


# BERT-based classifier: use [CLS] embedding for each choice
class BertForQA(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        bsz, num_choices, seq_len = input_ids.size()
        input_ids = input_ids.view(bsz * num_choices, seq_len)
        attention_mask = attention_mask.view(bsz * num_choices, seq_len)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # take [CLS]
        logits = self.classifier(cls_emb)
        return logits.view(bsz, num_choices)


# standard fine-tuning loop
def train(model, dataloader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss, total_acc = 0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = classifier_accuracy(logits.detach().cpu(), labels.cpu())
        total_loss += loss.item()
        total_acc += acc
    return total_loss / len(dataloader), total_acc / len(dataloader)


# validation loop (no gradient)
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            acc = classifier_accuracy(logits.cpu(), labels.cpu())
            total_loss += loss.item()
            total_acc += acc
    return total_loss / len(dataloader), total_acc / len(dataloader)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/obqa.train.txt")
    parser.add_argument("--valid_path", default="data/obqa.valid.txt")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Using device: {torch.cuda.get_device_name(0)}") # print gpu info

    train_set = OpenBookQADataset(args.train_path, tokenizer, args.max_length)
    val_set = OpenBookQADataset(args.valid_path, tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = BertForQA(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, scheduler, device, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/classifier.pt")
    print("Model saved to models/classifier.pt")
    # running time
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
