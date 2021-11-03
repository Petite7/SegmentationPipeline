import torch
import torch.nn as nn
from models.Segformer.mix_transformer import *
from models.Segformer.segformer_head import SegFormerHead
from models.Segformer.wrappers import resize

norm_cfg = dict(type='SyncBN', requires_grad=True)


class SegFormerNet(nn.Module):
    def __init__(self, num_classes, shape):
        super(SegFormerNet, self).__init__()
        self.shape = shape
        # mit_b2 - mit_b4 available
        self.backbone = mit_b5()
        self.decoder = SegFormerHead(in_channels=[64, 128, 320, 512],
                        in_index=[0, 1, 2, 3],
                        feature_strides=[4, 8, 16, 32],
                        channels=128,
                        dropout_ratio=0.1,
                        num_classes=num_classes,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        decoder_params=dict(embed_dim=768))

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        x = resize(
            input=x,
            size=self.shape,
            mode='bilinear',
            align_corners=False)
        return x
