# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 11:42
# @Author : 李昌杏
# @File : teacher.py
# @Software : PyCharm

"""
Acknowledgements:
1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
2. https://github.com/rishikksh20/CrossViT-pytorch
"""
import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from network.module import Attention, PreNorm, FeedForward, CrossAttention
from network.encoder import PhotoEncoder,SketchEncoder
class XMA(nn.Module):

    def __init__(self, dim=192, dim_head=64, cross_attn_depth=1, cross_attn_heads=3, dropout=0.):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
                PreNorm(dim,
                        CrossAttention(dim, heads=cross_attn_heads, dim_head=dim_head, dropout=dropout)),
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
                PreNorm(dim,
                        CrossAttention(dim, heads=cross_attn_heads, dim_head=dim_head, dropout=dropout)),
            ]))

    def forward(self, x_branch_1, x_branch_2):
        for f_12, g_21, cross_attn_s, f_21, g_12, cross_attn_l in self.cross_attn_layers:
            branch_1_class = x_branch_1[:, 0]
            x_branch_1 = x_branch_1[:, 1:]
            branch_2_class = x_branch_2[:, 0]
            x_branch_2 = x_branch_2[:, 1:]

            cal_q = f_21(branch_2_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_branch_1), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_12(cal_out)
            x_branch_2 = torch.cat((cal_out, x_branch_2), dim=1)

            cal_q = f_12(branch_1_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_branch_2), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_21(cal_out)
            x_branch_1 = torch.cat((cal_out, x_branch_1), dim=1)

        return x_branch_1, x_branch_2

class ModalityFusionNetwork(nn.Module):
    def __init__(self, feature_dim=768, cross_attn_depth=1,
                 enc_depth=3, heads=3,dropout=0.,num_class=104,
                 encoder_backbone='vit_base_patch16_224',checkpoint_path='vit.npz'):
        super().__init__()

        self.x_modal_transformers = nn.ModuleList([])
        for _ in range(enc_depth):
            self.x_modal_transformers.append(
                XMA(dim=feature_dim,
                    dim_head=feature_dim // heads,
                    cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                    dropout=dropout))

        self.mlp_head_skt = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_class)
        )

        self.mlp_head_img = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_class)
        )
        transformer_enc_branch_1 = PhotoEncoder (num_classes=125,feature_dim=feature_dim, encoder_backbone=encoder_backbone,checkpoint_path=checkpoint_path)
        transformer_enc_branch_2 = SketchEncoder(num_classes=125,feature_dim=feature_dim, encoder_backbone=encoder_backbone,checkpoint_path=checkpoint_path)

        self.transformer_enc_branch_1 = transformer_enc_branch_1
        self.transformer_enc_branch_2 = transformer_enc_branch_2

    def repr_branch_1(self, image):
        return  self.transformer_enc_branch_1.embedding(image)

    def repr_branch_2(self, image):
        return self.transformer_enc_branch_2.embedding(image)

    def cross_modal_embedding(self, x_branch_1, x_branch_2):
        for x_modal_transformer in self.x_modal_transformers:
            x_branch_1, x_branch_2 = x_modal_transformer(x_branch_1, x_branch_2)

        return x_branch_1[:, 0],x_branch_2[:, 0],x_branch_1[:, 1:], x_branch_2[:, 1:]

    def forward(self, image_1, image_2):
        x_branch_1 = self.repr_branch_1(image_1)
        x_branch_2 = self.repr_branch_2(image_2)
        photo1_cls, sketch1_cls, photo1_fea, sketch1_fea=self.cross_modal_embedding(x_branch_1, x_branch_2)

        return photo1_cls, sketch1_cls, self.mlp_head_img(photo1_cls),self.mlp_head_skt(sketch1_cls)
