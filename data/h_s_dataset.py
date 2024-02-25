import glob
import torch
import numpy as np
import os.path as op
from torch.utils.data import Dataset
import pickle
import os

class HSDataset(Dataset):
    def __init__(self,mode,partial_dataset=1,device='cuda',
                 data_file="/root/autodl-tmp/hate_speech_dataset/processed/dataset_features.pkl" ,
                 splits_dir="/root/autodl-tmp/hate_speech_dataset/splits"):
        self.mode = mode
        self.device = device
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'."
        self.partial_dataset = partial_dataset if mode != 'test' else 1
        # 加载所有数据
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # 加载对应模式的ID
        split_file_path = os.path.join(splits_dir, f"{mode}_ids.txt")
        with open(split_file_path, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        if self.partial_dataset < 1:
            reduce_size = int(len(self.ids) * self.partial_dataset)
            self.ids = self.ids[:reduce_size]

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # 直接从加载的数据中获取样本
        sample = self.data.get(img_id, None)
        if sample is None:
            print(f"Data for ID {img_id} not found.")
            return None

        image_features = torch.from_numpy(sample['image_features'])
        text_features = torch.from_numpy(sample['text_features'])
        lst = sample['labels']
        count_1 = np.count_nonzero(lst)
        count_0 = len(lst) - count_1        
        if count_0 > count_1:
            labels = torch.tensor(0)
        else:
            labels = torch.tensor(1)
        
        
        return {"image":image_features, "text":text_features, "label":labels}