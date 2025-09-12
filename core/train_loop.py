import os
import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------- small helpers ----------
def _bool_arg(args, name, default=False):
    return bool(getattr(args, name, default))

# ---------------- Logging ----------------
def log_epoch(epoch: int, metrics: dict, epoch_time: float):
    print(f"\n📊 Epoch {epoch:02d} Summary:")
    print(f"🧠 Train Loss: {metrics['train_loss']:.4f}")
    print(f"✅ Val Loss: {metrics['val_loss']:.4f} | Acc: {metrics['val_acc']:.2f}% | "
          f"Precision: {metrics['val_precision']:.2f} | Recall: {metrics['val_recall']:.2f} | F1: {metrics['val_f1']:.2f}")

    cm = metrics['cm']
    n_classes = cm.shape[0]

    # Create a pandas DataFrame for nicer display
    df_cm = pd.DataFrame(cm, index=[f"Ground {i}" for i in range(n_classes)],
                         columns=[f"Pred {i}" for i in range(n_classes)])
    print("\n🧾 Confusion Matrix:")
    print(df_cm)

    print(f"\n⏱️ Epoch Time: {epoch_time//60:.0f} minutes\n")




def plot_ct_collage(batch_x, n_rows=8, n_cols=5):
    """
    batch_x: tensor of shape (B, 1, D, H, W)
    Plots the first sample in the batch as a collage of slices.
    """
    sample = batch_x[0, 0]  # shape (D, H, W)
    D, H, W = sample.shape
    assert D <= n_rows * n_cols, f"Not enough grid spaces for {D} slices"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(D):
        axes[i].imshow(sample[i].cpu().numpy(), cmap="gray")
        axes[i].axis("off")

    # hide any extra axes
    for j in range(D, n_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------- Training & Evaluation ----------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    if total == 0:
        return 0.0
    return running_loss / total



def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    loss_sum = 0.0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = F.cross_entropy(output, y)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            batch_size = x.size(0)
            loss_sum += loss.item() * batch_size
            total += batch_size

    if total == 0:
        # avoid division by zero
        return 0.0, 0.0, 0.0, 0.0, 0.0, np.zeros((1, 1), dtype=int)

    loss_avg = loss_sum / total
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))

    # Binary metrics (works for binary classification)
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



def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler, args):
    device = args.device
    model.to(device)


    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = Path("best_model.pt")
    # ensure parent dir exists
    if best_model_path.parent and not best_model_path.parent.exists():
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Early stopping setup
    patience = getattr(args, "early_stopping", None)
    if patience is None:
        patience = 0
    try:
        patience = int(patience)
    except Exception:
        patience = 0

    epochs_no_improve = 0
    early_stop_triggered = False

    for epoch in range(args.epochs):

        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nStarted epoch {epoch} - Learning Rate: {current_lr:.6f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # run validation
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

        # NOTE: many schedulers expect step AFTER validation; if you're using ReduceLROnPlateau,
        # call scheduler.step(val_loss) instead of scheduler.step().
        try:
            # safe call in case scheduler API differs
            if hasattr(scheduler, "step") and scheduler is not None:
                # Example: if ReduceLROnPlateau use scheduler.step(val_loss)
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        except Exception as e:
            print("Scheduler step failed:", e)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_best_model(model, best_model_path)
            print(f"💾 Best model updated at epoch {epoch} (val_loss: {val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"— no improvement for {epochs_no_improve} epoch(s) (patience={patience})")

        # Early stopping decision (patience > 0 enables the feature)
        if patience > 0 and epochs_no_improve >= patience:
            print(f"⏸️ Early stopping triggered. No improvement for {epochs_no_improve} epochs (patience={patience}).")
            early_stop_triggered = True
            break

    if not best_model_path.exists():
        raise RuntimeError("No best model was saved during training. Check training loop.")

    # === Evaluate best model on test set ===
    map_location = device if isinstance(device, str) else ("cpu" if device.type == "cpu" else None)
    if map_location == None:
        state = torch.load(best_model_path)
    else:
        state = torch.load(best_model_path, map_location=map_location)

    model.load_state_dict(state)
    model.to(device)
    print(f"\n🧪 Test Evaluation (Best Model from Epoch {best_epoch}):")
    test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = evaluate(model, test_loader, device)
    print(f"📈 Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Precision: {test_prec:.2f} | Recall: {test_rec:.2f} | F1: {test_f1:.2f}")
    print("🧾 Confusion Matrix:\n", test_cm)

    # === Save test results + args + timestamp ===
    results_csv = Path("test_results.csv")
    write_header = not results_csv.exists()

    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": best_epoch,
        "test_acc": f"{test_acc:.2f}",
        "test_loss": f"{test_loss:.4f}",
        "test_precision": f"{test_prec:.2f}",
        "test_recall": f"{test_rec:.2f}",
        "test_f1": f"{test_f1:.2f}",
        "early_stopping_patience": patience,
        "early_stopping_triggered": int(early_stop_triggered),
        **vars(args)  # flatten argparse.Namespace into dict
    }

    # append row to CSV
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
