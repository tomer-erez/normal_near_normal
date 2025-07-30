from core.args import parse_args
from core.build import build_optimizer_scheduler, maybe_resume,generate_data_splits
from core.deep_models import CTTextCrossAttentionModel
from core.image_data_loader import DummyCTTextDataset
from core.train import train_and_evaluate
from transformers import AutoTokenizer
import torch

def main():
    args = parse_args()
    print("Arguments:", vars(args))

    # Model
    model = CTTextCrossAttentionModel()
    model.to(args.device)

    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    full_dataset = DummyCTTextDataset(tokenizer)

    # Split into train / val / test
    train_loader,val_loader,test_loader = generate_data_splits(args, full_dataset)

    # Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_scheduler(model, args.lr, args.weight_decay, args.epochs)
    model, optimizer, start_epoch = maybe_resume(model, optimizer, args.resume)

    train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler, args, start_epoch)

if __name__ == '__main__':
    main()
