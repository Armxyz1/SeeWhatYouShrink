import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.7, temperature=4.0):
    # Soft targets (distillation)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard targets (ground truth)
    ce_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * kd_loss + (1 - alpha) * ce_loss

def train_one_epoch_distillation(student_model, teacher_model, loader, optimizer, device, alpha=0.7, temperature=4.0):
    student_model.train()
    teacher_model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.squeeze().long().to(device)

        optimizer.zero_grad()
        student_logits = student_model(x)
        with torch.no_grad():
            teacher_logits = teacher_model(x)

        loss = distillation_loss(student_logits, teacher_logits, y, alpha, temperature)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(student_logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def evaluate_distillation(student_model, teacher_model, loader, device):
    student_model.eval()
    teacher_model.eval()
    correct, total = 0, 0
    running_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.squeeze().long().to(device)
            student_logits = student_model(x)
            teacher_logits = teacher_model(x)

            loss = distillation_loss(student_logits, teacher_logits, y)
            running_loss += loss.item()

            preds = torch.argmax(student_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def run_training_distillation(student_model, teacher_model, train_loader, val_loader, optimizer, device, alpha=0.7, temperature=4.0, epochs=10, 
                              scheduler=None, save_path='checkpoint.pth', resume=False, patience=3):
    best_acc = 0.0
    start_epoch = 0
    patience_counter = 0

    if resume and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        student_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        train_loss, train_acc = train_one_epoch_distillation(student_model, teacher_model, train_loader, optimizer, device, alpha, temperature)
        val_loss, val_acc = evaluate_distillation(student_model, teacher_model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({'model': student_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, save_path)
            print(f"New best model saved with accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return best_acc
