import os
from DatasetParse import data_spilt, ReadH5Pkld, get_loader
import torch
from TrainInference import inference_net_heatmap
from monai.networks.nets.swin_unetr import SwinUNETR


base_dir = '/raid/candi/xiangcen/data_all_modality_update_crop_three'
model_path = '/home/xiangcen/PathDetection/models/t2_adc_dwi_no_rad_407.ptm'
train_list, inference_list = data_spilt(base_dir, 725, 407)
device = 'cuda:1'

print(len(train_list), len(inference_list))



inference_transform = ReadH5Pkld()



inference_loader = get_loader(inference_list, inference_transform, batch_size=1, shuffle=False, drop_last=False)






model = SwinUNETR(img_size=(128, 128, 64), in_channels=3, out_channels=3, feature_size= 24, drop_rate = 0.1,
    attn_drop_rate = 0.1, downsample='mergingv2', use_v2=True)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

inference_net_heatmap(
        model,
        inference_loader,
        device=device,
        pirads_sheet = '/home/xiangcen/PathDetection/Target-Data_2019-12-05-2.xlsx',
        no_rad=True
    )
