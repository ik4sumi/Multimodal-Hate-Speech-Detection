import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class HateSpeechDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化数据集。
        :param root_dir: 数据集的根目录。
        :param split: 'train', 'val', 或 'test'。
        :param transform: 应用于图像的转换。
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # 加载标注数据
        with open(os.path.join(root_dir, 'MMHS150K_GT.json'), 'r') as file:
            self.annotations = json.load(file)
        
        # 加载分割数据
        split_file = os.path.join(root_dir, 'splits', f'{split}_ids.txt')
        with open(split_file, 'r') as file:
            self.ids = [line.strip() for line in file.readlines()]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        tweet_id = self.ids[idx]
        annotation = self.annotations[tweet_id]
        
        # 加载图像
        img_path = os.path.join(self.root_dir, 'img_resized', tweet_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 加载文本
        txt_path = os.path.join(self.root_dir, 'img_txt', tweet_id + '.txt')
        with open(txt_path, 'r') as file:
            text = file.read().strip()
        
        # 获取标签（取三个标注者标签的模式，如果不存在模式则取第一个标注者的标签）
        labels = torch.tensor(annotation['labels'])
        label = torch.mode(labels).values.item() if labels.unique().size(0) > 1 else labels[0].item()
        
        return image, text, label