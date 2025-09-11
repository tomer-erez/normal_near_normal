import os
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+


def build_optimizer_scheduler(model, lr, weight_decay, total_epochs):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    return optimizer, scheduler

def maybe_resume(model, optimizer, resume_path):
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")
        return model, optimizer, start_epoch
    return model, optimizer, 0

def generate_loaders(args,full_dataset):
    n = len(full_dataset)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    return train_loader,val_loader,test_loader


def setup_output_file(args):

    log_path = args.log_path

    # Optional: remove file if you want, but opening in 'w' is enough
    if os.path.exists(log_path):
        os.remove(log_path)

    # Open file in write mode with line buffering
    log_file = open(log_path, "w", buffering=1)

    # Redirect stdout
    sys.stdout = log_file


    # Get current time in Israel
    israel_time = datetime.now(ZoneInfo("Asia/Jerusalem"))

    # Print in a readable format
    print("Current Israel time:", israel_time.strftime("%Y-%m-%d %H:%M:%S"))
    print()  # extra newline

    for key, value in vars(args).items():
        print(f"{key}: {value}")  # \n adds an extra newline between entries

