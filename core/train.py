import time
import torch
import torch.nn.functional as F
import numpy as np
import os
import csv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def log_epoch(epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, cm_norm, epoch_time):
    print(f"\n📊 Epoch {epoch:02d} Summary:")
    print(f"🧠 Train Loss: {train_loss:.4f}")
    print(f"✅ Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Precision: {val_precision:.2f} | Recall: {val_recall:.2f} | F1: {val_f1:.2f}")
    print("📉 Normalized Confusion Matrix (rows = true labels):")
    print(np.round(cm_norm, 3))
    print(f"⏱️  Epoch time: {epoch_time:.2f} sec")


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for image, input_ids, attn_mask, label in dataloader:
        image, input_ids, attn_mask, label = image.to(device), input_ids.to(device), attn_mask.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image, input_ids, attn_mask)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, input_ids, attn_mask, label in dataloader:
            image, input_ids, attn_mask, label = image.to(device), input_ids.to(device), attn_mask.to(device), label.to(device)
            output = model(image, input_ids, attn_mask)
            loss = F.cross_entropy(output, label)
            preds = output.argmax(dim=1)

            correct += (preds == label).sum().item()
            loss_sum += loss.item() * image.size(0)
            total += image.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    acc = 100.0 * correct / total
    loss_avg = loss_sum / total

    # Confusion matrix (normalized)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Binary classification: class 2 = positive, classes 0+1 = negative
    bin_preds = [1 if p == 2 else 0 for p in all_preds]
    bin_labels = [1 if l == 2 else 0 for l in all_labels]
    precision = precision_score(bin_labels, bin_preds, zero_division=0)
    recall = recall_score(bin_labels, bin_preds, zero_division=0)
    f1 = f1_score(bin_labels, bin_preds, zero_division=0)

    return loss_avg, acc, precision, recall, f1, cm_norm


def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler, args, start_epoch=0):
    device = args.device
    model.to(device)

    best_val_loss = float("inf")
    best_checkpoint_path = "best_model.pt"

    for epoch in range(start_epoch, args.epochs):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, cm_norm = evaluate(model, val_loader, device)

        epoch_time = time.time() - start
        log_epoch(epoch, train_loss, val_loss, val_acc, val_precision, val_recall, val_f1, cm_norm, epoch_time)

        scheduler.step()

        # Save regular checkpoint
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, f'checkpoint_epoch_{epoch}.pt')

        # Save best model if this val loss is lowest so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            best_epoch = epoch
            best_flag = True
        else:
            best_flag = False

    # === Load best model for final test ===
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.to(device)

    print(f"\n🧪 Final Evaluation on Test Set (Best Model from Epoch {best_epoch}):")
    test_loss, test_acc, test_precision, test_recall, test_f1, test_cm_norm = evaluate(model, test_loader, device)
    print(f"📈 Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1: {test_f1:.2f}")
    print("🧾 Normalized Confusion Matrix:\n", np.round(test_cm_norm, 3))

    # === Save test results only if this was the best run ===
    csv_path = "test_results.csv"
    write_headers = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_headers:
            writer.writerow(["epoch", "test_acc", "test_loss", "test_precision", "test_recall", "test_f1"])
        writer.writerow([
            best_epoch,
            f"{test_acc:.2f}",
            f"{test_loss:.4f}",
            f"{test_precision:.2f}",
            f"{test_recall:.2f}",
            f"{test_f1:.2f}"
        ])
