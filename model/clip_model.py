import torch
import torch.nn as nn
import clip
import timm
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class ClipModel(nn.Module):
    def __init__(self, clip_model_name='ViT-B/32', num_classes=1,
                 vision_model_name='swin_small_patch4_window7_224',
                 text_model_name='bert-base-uncased',
                 freeze=False,
                 image_only=False,
                 text_only=False):
        super(ClipModel, self).__init__()
        #self.clip_model, _ = clip.load(clip_model_name, jit=False)
        #self.clip_model = self.clip_model.half()
        #self.clip_model = self.clip_model.to(dtype=torch.float16)
        
        #for param in self.clip_model.parameters():
            #param.requires_grad = True
        
        #text_features_dim = self.clip_model.text_projection.shape[1]
        #image_features_dim = self.clip_model.visual.output_dim
        self.vision_model = timm.create_model(vision_model_name, pretrained=True,global_pool='avg')
        self.text_model = BertModel.from_pretrained(text_model_name)

        text_features_dim = self.text_model.config.hidden_size
        vision_features_dim = self.vision_model.num_features

        self.image_only = image_only
        self.text_only = text_only
        
        if image_only and not text_only:
            self.in_channel = vision_features_dim
        elif text_only and not image_only:
            self.in_channel = text_features_dim
        else:
            self.in_channel = vision_features_dim + text_features_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.in_channel, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

            for name, param in self.vision_model.named_parameters():
                if name.startswith('layers.3') or name.startswith('head'):
                    param.requires_grad = True

            for param in self.text_model.parameters():
                param.requires_grad = False

            for name, param in self.text_model.named_parameters():
                if 'encoder.layer.11' in name or 'pooler' in name:
                    param.requires_grad = True
        
    def forward(self, input):

        images = input["image"]
        texts = input["text"]
        texts['input_ids'] = torch.squeeze(texts['input_ids'], dim=1)
        texts['attention_mask'] = torch.squeeze(texts['attention_mask'], dim=1)

        # 处理文本
        text_features = self.text_model(input_ids=texts['input_ids'],attention_mask=texts['attention_mask'])[1]  # 取[CLS] token的输出
        
        # 处理图像
        image_features = self.vision_model.forward_features(images)  # 获取图像特征
        image_features = image_features.permute(0,3,1,2)
        image_features = F.adaptive_avg_pool2d(image_features, (1, 1))
        image_features = image_features.squeeze(-1).squeeze(-1)

        #image_features = self.clip_model.encode_image(images)
        #text_features = self.clip_model.encode_text(texts)

        if self.image_only and not self.text_only:
            features = image_features
        elif self.text_only and not self.image_only:
            features = text_features
        else:
            features = torch.cat((image_features, text_features), dim=1)


        logits = self.classifier(features)

        return logits