from core import arguments
from core import utilities_module
from core.deep_models import *
from core import image_data_set
from core import train_loop

def choose_model(args):
    if args.model_choice == "Tiny3DTransformer":
        return Tiny3DTransformer(args).to(args.device)
    elif args.model_choice == "Tiny3DCNN":
        return Tiny3DCNN(args).to(args.device)

def main():
    print("starting training")
    args = arguments.get_args()
    utilities_module.setup_output_file(args)

    model = choose_model(args)
    ds = image_data_set.SinoCTDataset(args)
    train_loader,val_loader,test_loader=utilities_module.generate_loaders(args,ds)

    optimizer, scheduler = utilities_module.build_optimizer_scheduler(
        model, 
        args.lr, 
        args.weight_decay, 
        args.epochs
    )

    train_loop.train_and_evaluate(
        model, 
        train_loader, 
        val_loader, 
        test_loader, 
        optimizer, 
        scheduler, 
        args
    )

if __name__ == '__main__':
    main()
