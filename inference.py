import os
from DatasetParse import data_spilt, ReadH5Pkld, get_loader
from TrainInference import train_net, inference_net_lesion
from monai.transforms import *

from monai.losses import DiceLoss, FocalLoss, DiceFocalLoss
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets import DynUNet
from torch.optim import AdamW

import torch
import logging
import argparse



# base_dir = '/raid/candi/xiangcen/data_all_modality__update_crop'
# train_list, inference_list = data_spilt(base_dir, 745)

# Define available networks
nets = ['swin', 'dynunet']
datasets = ['t2_adc', 't2_adc_dwi', 't2', 't2_dwi', 't2_adc_dwi_no_rad']
# Set up argument parser
parser = argparse.ArgumentParser(description="Choose a network for training.")
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",  # Default to 'cuda:0'
    help="Choose the device (e.g., 'cuda:0', 'cuda:1', 'cpu')"
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Get trained model"
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=datasets,
    default=datasets[0],  # Default to 'swin'
    help=f"Choose a dataset from {datasets}"
)
parser.add_argument(
    "--seed",
    type=int,
    default=325,  # Default to 'swin'
    help=f"Choose a seed to generate dataset"
)


# Parse arguments
args = parser.parse_args()
device = args.device
model_path = args.model_path
dataset = args.dataset
seed = args.seed



t2_only, t2_dwi, no_rad = False, False, False
if dataset == 't2_adc':
    base_dir = '/raid/candi/xiangcen/data_all_modality__update_crop'
    # t2_only=False
    # t2_dwi=False
    input_channel = 3
elif dataset == 't2_adc_dwi':
    input_channel = 4
    base_dir = '/raid/candi/xiangcen/data_all_modality_update_crop_three'
    # t2_only=False
    # t2_dwi=False
elif dataset == 't2_dwi':
    input_channel = 3
    base_dir = '/raid/candi/xiangcen/data_all_modality_update_crop_three'
    # t2_only=False
    t2_dwi=True
elif dataset == 't2':
    input_channel = 2
    base_dir = '/raid/candi/xiangcen/data_all_modality__update_crop'
    t2_only=True
    # t2_dwi=False
elif dataset == 't2_adc_dwi_no_rad':
    input_channel = 3
    base_dir = '/raid/candi/xiangcen/data_all_modality_update_crop_three'
    no_rad=True
train_list, inference_list = data_spilt(base_dir, 725, seed)





inference_transform = ReadH5Pkld()



inference_loader = get_loader(inference_list, inference_transform, batch_size=1, shuffle=False, drop_last=False)



model = SwinUNETR(img_size=(128, 128, 64), in_channels=input_channel, out_channels=3, feature_size= 24, drop_rate = 0.1,
    attn_drop_rate = 0.1, downsample='mergingv2', use_v2=True)
model.load_state_dict(torch.load(model_path, map_location=device))



# tp_rate, fp_rate = inference_net(
#     model=model,
#     inference_loader=inference_loader,
#     device=device
# )

# print(f"TP Rate: {tp_rate}, FP Rate: {fp_rate}")



inference_net_lesion(
    model=model,
    inference_loader=inference_loader,
    device=device,
    pirads_sheet='Target-Data_2019-12-05-2.xlsx',
    t2_only=t2_only,
    t2_dwi=t2_dwi,
    no_rad=no_rad
)