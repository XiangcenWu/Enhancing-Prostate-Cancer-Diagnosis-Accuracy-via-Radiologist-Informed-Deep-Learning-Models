import os
import random
import h5py
import torch
from monai.data import Dataset, DataLoader




def data_spilt(base_dir, num_train, seed=325):
    list = os.listdir(base_dir)
    list = [os.path.join(base_dir, item) for item in list]
    random.seed(seed)
    random.shuffle(list)
    return list[:num_train], list[num_train:]


def load_h5_file(file_dir):
    
    h5f = h5py.File(file_dir, 'r')


    mri = torch.from_numpy(h5f['img'][:])
    contour = torch.from_numpy(h5f['lesion'][:])
    patient_name = h5f['patient_name'][()].decode('utf-8')
    patient_uid = h5f['patient_uid'][()].decode('utf-8')
    
    
    h5f.close()
    
    
    return mri, contour, patient_name, patient_uid


class ReadH5Pkld():
    def __init__(self):
        super().__init__()

    def __call__(self, file_dir):
        img, label, patient_name, patient_uid = load_h5_file(file_dir)
        return {
            'img': img,
            'label': label,
            'patient_name': patient_name,
            'patient_uid': patient_uid
        }
        
        

def get_loader(
        list, 
        transform, 
        batch_size: int,
        shuffle: bool=True, 
        drop_last: bool=False, 
    ):
    _ds = Dataset(list, transform=transform)

    return DataLoader(
        dataset = _ds,
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )
