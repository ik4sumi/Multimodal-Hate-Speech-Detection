import torch
import torch.nn as nn
import clip
import timm
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from einops import rearrange

class CrossModalAttention(nn.Module):
    def __init__(self, clip_model_name='ViT-B/32', num_classes=1,
                  vision_model_name='swin_small_patch4_window7_224',
                  text_model_name='bert-base-uncased',
                  freeze=False,
                  hidden_dims=128,
                  text_only=False,
                  image_only=False,
                  t_len=50,
                  v_len=49):
        super(CrossModalAttention, self).__init__()

        # swin transformer & bert
        self.vision_model = timm.create_model(vision_model_name, pretrained=True,global_pool='avg')
        self.text_model = BertModel.from_pretrained(text_model_name)

        text_features_dim = self.text_model.config.hidden_size
        vision_features_dim = self.vision_model.num_features

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

        # cross-attention modules      
        self.v2t = CrossAttentionModule(vision_features_dim,text_features_dim,hidden_dims,x_len=v_len,y_len=t_len)
        self.t2v = CrossAttentionModule(text_features_dim,vision_features_dim,hidden_dims,x_len=t_len,y_len=v_len)

        self.text_only = text_only
        self.image_only = image_only

        if not text_only and not image_only:
            hidden_dims *=2

        # output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims, 2048),
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

        
    def forward(self, input):

        images = input["image"]
        texts = input["text"]
        texts['input_ids'] = torch.squeeze(texts['input_ids'], dim=1)
        texts['attention_mask'] = torch.squeeze(texts['attention_mask'], dim=1)

        # 处理文本
        text_features = self.text_model(input_ids=texts['input_ids'],attention_mask=texts['attention_mask'])['last_hidden_state']
        
        # 处理图像
        image_features = self.vision_model.forward_features(images)  # 获取图像特征
        image_features = rearrange(image_features, 'b h w d -> b (h w) d')


        v2t = self.v2t(image_features,text_features)
        t2v = self.t2v(text_features,image_features)

        if self.text_only and not self.image_only:
            features = v2t
        elif self.image_only and not self.text_only:
            features = t2v
        else:
            features = torch.cat((v2t, t2v), dim=1)

        logits = self.classifier(features)

        return logits

class CrossAttentionModule(nn.Module):
    def __init__(self, x_dim, y_dim, attention_dim=128,dropout_rate=0.1,x_len=1,y_len=1):
        super(CrossAttentionModule, self).__init__()

        self.query_transform = nn.Linear(x_dim, attention_dim)
        self.key_transform = nn.Linear(y_dim, attention_dim)
        self.value_transform = nn.Linear(y_dim, attention_dim)
        self.attention_dim = attention_dim
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_dim * 4, attention_dim)
        )
        self.normx = nn.LayerNorm(x_dim)
        self.normy = nn.LayerNorm(y_dim)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.scoring_layer = nn.Linear(attention_dim, 1)

        self.positional_encoding_x = nn.Parameter(torch.randn(x_len, x_dim))
        self.positional_encoding_y = nn.Parameter(torch.randn(y_len, y_dim))

    def forward(self, x, y):
        """
        tensor_a, tensor_b 的形状: (b, n, d)
        """
        x = self.normx(x)
        y = self.normy(y)

        x = x + self.positional_encoding_x
        y = y + self.positional_encoding_y

        query = self.query_transform(x)  # (b, n, attention_dim)
        key = self.key_transform(y)      # (b, n, attention_dim)
        value = self.value_transform(y)  # (b, n, attention_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (b, n, n)
        attention_scores = attention_scores / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)       # (b, n, n)
        attention_weights = self.attention_dropout(attention_weights)

        # 应用注意力权重到 value 上
        y = torch.matmul(attention_weights, value)  # (b, n, attention_dim)

        y = self.norm1(y + self.dropout1(y))
        ff_output = self.feed_forward(y)
        y = self.norm2(y + self.dropout2(ff_output))

        scores = self.scoring_layer(y)
        weights = F.softmax(scores, dim=1)
        weighted_y = torch.sum(weights * y, dim=1)

        return weighted_y