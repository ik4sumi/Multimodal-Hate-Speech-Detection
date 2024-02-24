import glob
import torch
import numpy as np
import os.path as op
from torch.utils.data import Dataset

class HSDataset(Dataset):
    def __init__(self,mode,partial_dataset=1,device='cuda',
                 img_dir="/root/autodl-tmp/hate_speech_dataset/processed/img",
                 text_dir="/root/autodl-tmp/hate_speech_dataset/processed/txt",
                 label_dir="/root/autodl-tmp/hate_speech_dataset/processed/label",
                 splits_dir="/root/autodl-tmp/hate_speech_dataset/splits"):
        self.mode = mode
        self.device = device
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'."
        self.partial_dataset = partial_dataset if mode != 'test' else 1
        img_paths = glob.glob(op.join(img_dir, '*.pt'), recursive=True)
        text_paths = glob.glob(op.join(text_dir, '*.pt'), recursive=True)
        label_paths = glob.glob(op.join(label_dir, '*.pt'), recursive=True)

        split_file_path = op.join(splits_dir, f"{mode}_ids.txt")
        with open(split_file_path, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        if self.partial_dataset < 1:
            reduce_size = int(len(self.ids) * self.partial_dataset)
            self.ids = self.ids[:reduce_size]

        self.img_paths = {op.basename(path).split('_')[0]: path for path in glob.glob(op.join(img_dir, '*.pt'))}
        self.text_paths = {op.basename(path).split('_')[0]: path for path in glob.glob(op.join(text_dir, '*.pt'))}
        self.label_paths = {op.basename(path).split('_')[0]: path for path in glob.glob(op.join(label_dir, '*.pt'))}

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        img_path = self.img_paths.get(img_id, None)
        text_path = self.text_paths.get(img_id, None)
        label_path = self.label_paths.get(img_id, None)
        
        if img_path:
            image = torch.load(img_path).to(self.device)
        else:
            print(f"Image for ID {img_id} not found.")
            image = None
        
        if text_path:
            text = torch.load(text_path).to(self.device)
        else:
            print(f"Text for ID {img_id} not found.")
            text = None
        
        if label_path:
            label = torch.load(label_path).to(self.device)
        else:
            print(f"Label for ID {img_id} not found.")
            label = None
        
        return image, text, label