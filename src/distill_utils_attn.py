import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

class FeatureProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

def distillation_loss(student_logits, teacher_logits, student_features, teacher_features, labels,
                      temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2, projector=None):
    # KL divergence between softened logits
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Cross-entropy with ground truth
    ce_loss = F.cross_entropy(student_logits, labels)

    # MSE between selected intermediate features (after projection)
    feature_loss = 0.0
    for sf, tf in zip(student_features, teacher_features):
        if projector is not None:
            sf = projector(sf)
        feature_loss += F.mse_loss(sf, tf)

    total_loss = alpha * kd_loss + beta * feature_loss + gamma * ce_loss
    return total_loss

def train_one_epoch_distillation(student_model, teacher_model, loader, optimizer, device,
                                 temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2, projector=None):
    student_model.train()
    teacher_model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.squeeze().long().to(device)
        optimizer.zero_grad()

        student_logits, student_features = student_model(x, return_features=True)
        with torch.no_grad():
            teacher_logits, teacher_features = teacher_model(x, return_features=True)

        loss = distillation_loss(
            student_logits, teacher_logits,
            student_features, teacher_features,
            y, temperature, alpha, beta, gamma, projector
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = student_logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def evaluate_distillation(student_model, teacher_model, loader, device,
                          temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2, projector=None):
    student_model.eval()
    teacher_model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.squeeze().long().to(device)
            student_logits, student_features = student_model(x, return_features=True)
            teacher_logits, teacher_features = teacher_model(x, return_features=True)

            loss = distillation_loss(
                student_logits, teacher_logits,
                student_features, teacher_features,
                y, temperature, alpha, beta, gamma, projector
            )

            running_loss += loss.item()
            preds = student_logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def run_training_distillation(student_model, teacher_model, train_loader, val_loader, optimizer, device,
                              temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2, projector=None,
                              epochs=10, scheduler=None, save_path='checkpoint.pth', resume=False, patience=3):
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
        train_loss, train_acc = train_one_epoch_distillation(
            student_model, teacher_model, train_loader, optimizer, device,
            temperature, alpha, beta, gamma, projector
        )
        val_loss, val_acc = evaluate_distillation(
            student_model, teacher_model, val_loader, device,
            temperature, alpha, beta, gamma, projector
        )

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({'model': student_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, save_path)
            print(f"New best model saved with acc: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return best_acc
