import torch
import clip
import json
from PIL import Image
import os
from tqdm import tqdm
import pickle
import numpy as np
from torchvision import transforms
import timm
from transformers import RobertaModel, RobertaTokenizer
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
import torch.nn.functional as F

# 定义数据和文件路径
img_dir = "/root/autodl-tmp/hate_speech_dataset/img_resized"
json_file = "/root/autodl-tmp/hate_speech_dataset/MMHS150K_GT.json"
img_txt_dir = "/root/autodl-tmp/hate_speech_dataset/img_txt"
train_ids_file = "splits/train_ids.txt"
val_ids_file = "splits/val_ids.txt"
test_ids_file = "splits/test_ids.txt"
img_save_dir = "/root/autodl-tmp/hate_speech_dataset/processed/img"
txt_save_dir = "/root/autodl-tmp/hate_speech_dataset/processed/txt"
label_save_dir = "/root/autodl-tmp/hate_speech_dataset/processed/label"
save_dir = "/root/autodl-tmp/hate_speech_dataset/processed"

# 加载预训练的CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
nltk.download('stopwords')

# 函数：加载并预处理图像
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)

# 函数：提取图像特征
def extract_image_features(image_path):
    image_input = load_and_preprocess_image(image_path)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features

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

def extract_text_features(model, text, device):
    """
    对长文本进行分割并提取特征。
    """
    segments = split_text(text)
    features_list = []
    
    for segment in segments:
        text_input = clip.tokenize([segment],truncate=True).to(device)
        with torch.no_grad():
            features = model.encode_text(text_input)
            features_list.append(features)
    
    # 将所有片段的特征进行平均，作为整个文本的特征表示
    features = torch.mean(torch.stack(features_list), dim=0)
    return features

# 加载 JSON 数据
with open(json_file, 'r') as f:
    data = json.load(f)

def extract_swin_features(image_path,device):
    # 加载预训练的Swin Transformer模型
    model = timm.create_model('swinv2_tiny_window8_256', pretrained=True).to(device).half()
    model.eval()  # 设置为评估模式

    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载并预处理图像
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = img_t.unsqueeze(0).to(device).half()  # 添加批处理维度

    # 提取特征
    with torch.no_grad():
        features = model(batch_t)

    return features

def extract_bert_features(text,device):
    segments = split_text(text)
    features_list = []
    target_seq_length = 50
    for segment in segments:
        # 初始化RoBERTa的tokenizer和model
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base').to(device).half()

        # 编码文本
        text = segment
        encoded_input = tokenizer(text, return_tensors='pt')

        # 不要对token IDs调用.half()
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        # 提取特征
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = output.last_hidden_state[:, :, :]  # 取[CLS]标记的特征
            # 调整特征序列到目标长度
            current_seq_length = text_features.shape[1]
            if current_seq_length < target_seq_length:
                # Padding
                padding_length = target_seq_length - current_seq_length
                padded_features = F.pad(text_features, (0, 0, 0, padding_length), "constant", 0)
                final_features = padded_features
            else:
                # 截断或者直接使用，根据实际长度决定
                final_features = text_features[:, :target_seq_length, :]
            features_list.append(final_features)

        # text_features是提取的文本特征
    


    # 将所有片段的特征进行平均，作为整个文本的特征表示
    features = torch.mean(torch.stack(features_list), dim=0)
    return features



# 函数：遍历图像和文本，提取特征
def process_dataset():
    dataset_features = {}

    for tweet_id, details in tqdm(data.items()):
        entry = {}
        img_path = os.path.join(img_dir, tweet_id + ".jpg")
        txt_path = os.path.join(img_txt_dir, tweet_id + ".json")

        # 处理图像文件
        if os.path.exists(img_path):
            #image_features = extract_image_features(img_path)
            image_features =  extract_swin_features(img_path, device)
            entry["image_features"] = image_features.cpu().numpy()
        else:
            print(f"Image file for {tweet_id} not found.")
        
        # 处理文本文件
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                d = json.load(f)
                text = d["img_text"] + details["tweet_text"]
        else:
            text = details["tweet_text"]
        
        #text_features = extract_text_features(model, text, device)
        text_features = extract_bert_features(text, device)
        entry["text_features"] = text_features.cpu().numpy()

        # 保存标签
        entry["labels"] = np.array(details["labels"])
        
        # 将条目保存到大字典中
        dataset_features[tweet_id] = entry

    # 使用pickle保存整个字典
    with open(os.path.join(save_dir, "dataset_features_swin_bert.pkl"), 'wb') as f:
        pickle.dump(dataset_features, f, protocol=pickle.HIGHEST_PROTOCOL)
# 运行数据处理
process_dataset()