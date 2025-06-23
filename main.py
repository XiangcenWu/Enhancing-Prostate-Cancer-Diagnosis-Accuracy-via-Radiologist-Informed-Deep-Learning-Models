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
    "--batchsize",
    type=int,
    default=16,  # Default batch size
    help="Set batch size for training"
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
batchsize = args.batchsize
dataset = args.dataset
seed = args.seed

log_name = dataset + '_' + str(seed)
logging.basicConfig(
    filename=os.path.join('logs', log_name+".txt"),  # Log file
    filemode="w",  # Overwrite if exists
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


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
train_list, inference_list = data_spilt(base_dir, 725, seed=seed)



train_transform = Compose([
        ReadH5Pkld(),
        RandAffined(['img','label'], spatial_size=(128, 128, 64), prob=0.25, shear_range=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), mode='nearest', padding_mode='zeros'),
        RandGaussianSmoothd(['img'], prob=0.25),
        RandGaussianNoised(['img'], prob=0.25, std=0.05),
        RandAdjustContrastd(['img'], prob=0.25, gamma=(0.5, 2.))
    ])

inference_transform = ReadH5Pkld()



train_loader = get_loader(train_list, train_transform, batch_size=batchsize)
inference_loader = get_loader(inference_list, inference_transform, batch_size=1, shuffle=False, drop_last=False)



model = SwinUNETR(img_size=(128, 128, 64), in_channels=input_channel, out_channels=3, feature_size= 24, drop_rate = 0.1,
    attn_drop_rate = 0.1, downsample='mergingv2', use_v2=True)




# loss_function = DiceFocalLoss(include_background=True, to_onehot_y=True, sigmoid=True,\
#         weight=[1, 1, 3], lambda_dice=1, lambda_focal=3)
loss_function = FocalLoss(include_background=True, to_onehot_y=True, weight=[0.5, 1, 3], use_softmax=True)
optimizer = AdamW(model.parameters(), lr=1e-6)


best_combined_rate = 0
for e in range(30):
    loss = train_net(
        model=model,
        train_loader=train_loader,
        train_optimizer=optimizer,
        train_loss=loss_function,
        device=device,
        t2_only=t2_only,
        t2_dwi=t2_dwi,
        no_rad=no_rad
    )
    
    logging.info(f"Epoch {e+1}, Loss: {loss}")
    
    
    
    tp_rate, fp_rate = inference_net_lesion(
        model=model,
        inference_loader=inference_loader,
        device=device,
        pirads_sheet='Target-Data_2019-12-05-2.xlsx',
        t2_only=t2_only,
        t2_dwi=t2_dwi,
        no_rad=no_rad
    )
    
    logging.info(f"Epoch {e+1}, TP Rate: {tp_rate}, FP Rate: {fp_rate}")
    
    
        