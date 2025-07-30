import torch
import torch.nn.functional as F
from report import log_epoch, log_time
import time

def evaluate(model, dataloader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for image, input_ids, attn_mask, label in dataloader:
            image, input_ids, attn_mask, label = image.to(device), input_ids.to(device), attn_mask.to(device), label.to(device)
            output = model(image, input_ids, attn_mask)
            loss = F.cross_entropy(output, label)
            preds = output.argmax(dim=1)
            correct += (preds == label).sum().item()
            loss_sum += loss.item() * image.size(0)
            total += image.size(0)
    return loss_sum / total, 100.0 * correct / total

def train_and_evaluate(model, dataloader, optimizer, scheduler, args, start_epoch=0):
    device = args.device
    model.to(device)

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
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

        scheduler.step()
        train_loss = running_loss / len(dataloader.dataset)
        val_loss, val_acc = evaluate(model, dataloader, device)

        log_epoch(epoch, train_loss, val_loss, val_acc)
        log_time(start)

        # Optional checkpoint saving
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, f'checkpoint_epoch_{epoch}.pt')
