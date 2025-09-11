import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='CT + Text Classification')
    
    ##################################################################
    # ---------------------General training stuff---------------------
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_gamma', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    ##################################################################
    # ---------------------CT pre-processing stuff--------------------
    parser.add_argument('--d_target', type=int, default=40, help="number of ct slices to pad/truncate to")
    parser.add_argument('--brain_window_function_center', type=int, default=40)
    parser.add_argument('--brain_window_function_width', type=int, default=80)
    ##################################################################
    # --------------------------Dataset stuff-------------------------
    parser.add_argument('--sinoCT_dataset_path', type=str, default=r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\stanford_data\ctsinogram\head_ct_dataset_anon", 
    help='in the sinoCT dataset,' \
    'path to the folder which holds: batch_0, batch_1, batch_2... etc. might be called head_ct_dataset_anon' \
    'each batch folder holds: series_i, series_i+1...' \
    'each series is a participant, ' \
    'each participant has a reconstructed_image folder.' \
    'that folder has a bunch of dcm images: image_0.dcm, image_1.dcm... where each one is a brain CT slice')
    parser.add_argument('--sinoCT_csv_path', type=str, default=r"C:\Users\tomer.erez\Desktop\code_projects\normal_near_normal\stanford_data\labels.csv", 
    help='in the sinoCT dataset,' \
    'path to the sinoCT labels csv')
    ##################################################################
    # --------------------------Logging stuff-------------------------
    parser.add_argument('--log_path', type=str, default=r'.\log_file.log')
    ##################################################################




    return parser.parse_args()
