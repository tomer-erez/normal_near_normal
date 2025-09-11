import os
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm


# ---------------- Logging ----------------
def log_epoch(epoch: int, metrics: dict, epoch_time: float):
    """
    Print summary for one epoch.
    """
    print(f"\n📊 Epoch {epoch:02d} Summary:")
    print(f"🧠 Train Loss: {metrics['train_loss']:.4f}")
    print(f"✅ Val Loss: {metrics['val_loss']:.4f} | Acc: {metrics['val_acc']:.2f}% | "
          f"Precision: {metrics['val_precision']:.2f} | Recall: {metrics['val_recall']:.2f} | F1: {metrics['val_f1']:.2f}")
    print("🧾 Confusion Matrix (rows=true, cols=pred):")
    print(metrics['cm'])
    print(f"⏱️ Epoch Time: {epoch_time:.2f} sec")


# ---------------- Training & Evaluation ----------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = F.cross_entropy(output, y)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            loss_sum += loss.item() * x.size(0)

    loss_avg = loss_sum / len(all_labels)
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))

    # Binary metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return loss_avg, acc, precision, recall, f1, cm


# ---------------- Checkpoint helpers ----------------
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, path)


def save_best_model(model, path):
    torch.save(model.state_dict(), path)


# ---------------- Training Loop ----------------
def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler, args):
    device = args.device
    model.to(device)

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = Path("best_model.pt")

    for epoch in range(args.epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n Epoch {epoch} - Learning Rate: {current_lr:.6f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = evaluate(model, val_loader, device)

        epoch_time = time.time() - start_time
        log_epoch(epoch, {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'cm': val_cm
        }, epoch_time)

        scheduler.step()


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_best_model(model, best_model_path)
            print(f"💾 Best model updated at epoch {epoch}")

    # === Evaluate best model on test set ===
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    print(f"\n🧪 Test Evaluation (Best Model from Epoch {best_epoch}):")
    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = evaluate(model, test_loader, device)
    print(f"📈 Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Precision: {test_prec:.2f} | Recall: {test_rec:.2f} | F1: {test_f1:.2f}")
    print("🧾 Confusion Matrix:\n", test_cm)

    # Save test results
    results_csv = Path("test_results.csv")
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "test_acc", "test_loss", "test_precision", "test_recall", "test_f1"])
        writer.writerow([best_epoch, f"{test_acc:.2f}", f"{test_loss:.4f}",
                         f"{test_prec:.2f}", f"{test_rec:.2f}", f"{test_f1:.2f}"])
