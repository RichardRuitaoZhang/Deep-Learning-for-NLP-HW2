import torch

# basic accuracy for classification
def classifier_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# evaluation for generative approach (exact match on A/B/C/D)
@torch.no_grad()
def evaluate_generator(model, dataloader, id_list, device):
    model.eval()
    correct, total = 0, 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        logits = model(input_ids, None)
        last_logits = logits[:, -1, :]
        choice_logits = last_logits[:, id_list]
        preds = torch.argmax(choice_logits, dim=-1)
        correct += (preds.cpu() == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else 0.0
