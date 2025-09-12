import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='CT + Text Classification')
    ##################################################################
    # ---------------------where to run code-------------------------
    parser.add_argument('--local', action='store_true', help="run locally ,not on newton")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    ##################################################################
    # ---------------------General training config---------------------
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint path to resume training')
    parser.add_argument('--seed', type=int, default=42)
    ##################################################################
    # ---------------------CT pre-processing stuff--------------------
    parser.add_argument('--d_target', type=int, default=40, help="number of ct slices to pad/truncate to")
    parser.add_argument('--brain_window_function_center', type=int, default=40)
    parser.add_argument('--brain_window_function_width', type=int, default=80)
    ##################################################################
    # --------------------------Dataset stuff-------------------------
    parser.add_argument('--sinoCT_dataset_path', type=str,
                        default=r"/home/tomer.erez/normal_near_normal/stanford_data/ct_dataset/ctsinogram/head_ct_dataset_anon",
                        help='in the sinoCT dataset,' \
                        'path to the folder which holds: batch_0, batch_1, batch_2... etc. might be called head_ct_dataset_anon' \
                        'each batch folder holds: series_i, series_i+1...' \
                        'each series is a participant, ' \
                        'each participant has a reconstructed_image folder.' \
                        'that folder has a bunch of dcm images: image_0.dcm, image_1.dcm... where each one is a brain CT slice')
    parser.add_argument('--sinoCT_csv_path', type=str, default=r"/home/tomer.erez/normal_near_normal/stanford_data/labels.csv",
                        help='in the sinoCT dataset,' \
                        'path to the sinoCT labels csv')
    ##################################################################
    #-------------------------model_choice____________________________
    parser.add_argument('--model_choice', type=str, default="Tiny3DCNN",choices=["Tiny3DCNN", "Tiny3DTransformer"],
                        help='which model to run')





    args = parser.parse_args()

    if args.local:
        args.sinoCT_dataset_path=r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\stanford_data\ctsinogram\head_ct_dataset_anon"
        args.sinoCT_csv_path=r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\stanford_data\labels.csv"

    return args