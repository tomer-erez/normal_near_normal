from args import parse_args
from build import build_optimizer_scheduler, maybe_resume
from deep_models import CTTextCrossAttentionModel
from text_image_loader import DummyCTTextDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from train import train_and_evaluate


def main():
    args = parse_args()
    print("Arguments:", vars(args))

    # Model
    model = CTTextCrossAttentionModel()
    model.to(args.device)

    # Tokenizer and Data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    dataset = DummyCTTextDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_scheduler(model, args.lr, args.weight_decay, args.epochs)
    model, optimizer, start_epoch = maybe_resume(model, optimizer, args.resume)

    # Placeholder train script
    train_and_evaluate(model, dataloader, optimizer, scheduler, args, start_epoch=start_epoch)

if __name__ == '__main__':
    main()
