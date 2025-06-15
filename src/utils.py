import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.squeeze().long().to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.squeeze().long().to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


def run_training(model, train_loader, val_loader, optimizer, device, epochs=10,
                 scheduler=None, save_path='checkpoint.pth', resume=False, patience=3):
    best_acc = 0.0
    start_epoch = 0
    patience_counter = 0

    if resume and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch} | Best Acc: {best_acc:.4f}")

    for epoch in range(start_epoch, epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, save_path)
            print(f"Checkpoint saved: {save_path} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
