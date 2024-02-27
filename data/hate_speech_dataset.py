import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
import torch.nn.functional as F
import clip
from transformers import BertModel, BertTokenizer

class HateSpeechDataset(Dataset):
    def __init__(self, root_dir="/root/autodl-tmp/hate_speech_dataset", split='train', transform=None, mode='train',partial_dataset=1,device='cuda',):
        """
        初始化数据集。
        :param root_dir: 数据集的根目录。
        :param split: 'train', 'val', 或 'test'。
        :param transform: 应用于图像的转换。
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mode = mode
        self.device = device
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'."
        self.partial_dataset = partial_dataset if mode != 'test' and mode !='val' else 1

        # 加载标注数据
        with open(os.path.join(root_dir, 'MMHS150K_GT.json'), 'r') as file:
            self.annotations = json.load(file)
        
        # 加载分割数据
        split_file = os.path.join(root_dir, 'splits', f'{mode}_ids.txt')
        with open(split_file, 'r') as file:
            self.ids = [line.strip() for line in file.readlines()]

        clip_model, self.preprocess = clip.load("ViT-B/32")
        #self.tknz = clip.tokenize
        self.tknz = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return int(self.partial_dataset*len(self.ids))
    
    def __getitem__(self, idx):
        tweet_id = self.ids[idx]
        annotation = np.array(self.annotations[tweet_id]['labels'])
        count_1 = np.count_nonzero(annotation)
        count_0 = len(annotation) - count_1        
        if count_0 > count_1:
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)
        
        # 加载图像
        img_path = os.path.join(self.root_dir, 'img_resized', tweet_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        #image = self.preprocess(image)
        
        # 加载文本
        txt_path = os.path.join(self.root_dir, 'img_txt', tweet_id + '.txt')
        # 处理文本文件
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                d = json.load(f)
                text = d["img_text"] + self.annotations[tweet_id]["tweet_text"]
        else:
            text = self.annotations[tweet_id]["tweet_text"]

        text = self.tknz(text, padding='max_length', truncation=True, max_length=50, return_tensors="pt")
        
        # 获取标签（取三个标注者标签的模式，如果不存在模式则取第一个标注者的标签）
        #labels = torch.tensor(annotation['labels'])
        #label = torch.mode(labels).values.item() if labels.unique().size(0) > 1 else labels[0].item()
        
        return {"image":image, "text":text, "label":label}

def preprocess_text(text: str) -> str:
    
    # Convert to lowercase
    text = text.lower()

    # Non-word character Removal
    text = text.replace('[^\w\s]', '')
    
    # Digits Removal
    text = text.replace('\d', '')

    
    # Remove punctuation
    PUNCT_TO_REMOVE = string.punctuation
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    
    
    # Remove stopwords
    STOPWORDS = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    
    
    # Stem words
    stemmer = SnowballStemmer(language='english')
#     wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
#     pos_tagged_text = nltk.pos_tag(text.split())
    text = " ".join([stemmer.stem(word) for word in text.split()])
    

    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Remove HTML tags
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r'', text)

    return text

# 函数：提取文本特征
def split_text(text, max_length=77):
    """
    将文本分割成不超过最大长度的片段。
    尝试在空格处分割以保持单词的完整性。
    """
    text = preprocess_text(text)
    words = text.split()
    segments = []
    current_segment = ""
    
    for word in words:
        if len(current_segment) + len(word) + 1 <= max_length:
            current_segment += " " + word if current_segment else word
        else:
            segments.append(current_segment)
            current_segment = word
    if current_segment:
        segments.append(current_segment)
    
    return segments