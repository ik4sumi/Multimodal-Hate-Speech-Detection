import torch
import torch.nn as nn
import clip

class ClipModel(nn.Module):
    def __init__(self, clip_model_name='ViT-B/32', num_classes=1):
        super(ClipModel, self).__init__()
        self.clip_model, _ = clip.load(clip_model_name, jit=False)
        #self.clip_model = self.clip_model.half()
        #self.clip_model = self.clip_model.to(dtype=torch.float16)
        for param in self.clip_model.parameters():
            param.requires_grad = True
        
        text_features_dim = self.clip_model.text_projection.shape[1]
        image_features_dim = self.clip_model.visual.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(image_features_dim + text_features_dim, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, input):

        images = input["image"]
        texts = input["text"]

        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)


        combined_features = torch.cat((image_features, text_features), dim=1)


        logits = self.classifier(combined_features)

        return logits