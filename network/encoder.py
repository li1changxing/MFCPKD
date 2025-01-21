# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 11:41
# @Author : 李昌杏
# @File : encoder.py
# @Software : PyCharm
from torch import nn
import timm
from timm.models.vision_transformer import VisionTransformer
import torch
class PhotoEncoder(nn.Module):
    def __init__(self, num_classes, feature_dim=768, encoder_backbone='vit_base_patch16_224',checkpoint_path='weights/vit_base_patch16_224.npz'):
        super().__init__()

        flag=True
        if checkpoint_path==None:
            flag=False
        self.num_classes = num_classes
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=False,checkpoint_path=checkpoint_path)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )

    def embedding(self, photo):
        x = self.encoder.patch_embed(photo)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward_features(self, photo):
        return self.encoder.forward_features(photo)

    def classify(self, features):
        return self.mlp_head(features)

    def forward(self, photo):
        return self.classify(self.forward_features(photo))

    def get_token(self):return self.encoder.cls_token

class SketchEncoder(nn.Module):
    def __init__(self, num_classes, feature_dim=768, encoder_backbone='vit_base_patch16_224',checkpoint_path='../vit.npz'):
        super().__init__()

        flag = True
        if checkpoint_path == None:
            flag = False
        self.num_classes = num_classes
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=False,checkpoint_path=checkpoint_path)
        self.scale = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, 3, 2, 1, bias=False),
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )

    def embedding(self, sketch):
        x = self.encoder.patch_embed(sketch)
        b,h_w,d=x.shape
        x1 = self.scale(sketch)
        x1 = x1.view(b,d,h_w).transpose(1, 2)
        x=(x+x1)/2

        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward_features(self, sketch):
        return self.encoder.forward_features(sketch)

    def classify(self, features):
        return self.mlp_head(features)

    def forward(self, sketch):
        return self.classify(self.forward_features(sketch))

    def get_token(self): return self.encoder.cls_token

if __name__ == '__main__':
    a=SketchEncoder(100).cuda()
    data = torch.rand(32,3,224,224).cuda()
    output = a.embedding(data)
    print(output.shape)
