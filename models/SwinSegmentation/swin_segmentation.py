import torch
import torch.nn as nn
import numpy as np
from models.SwinSegmentation.swin_transformer import SwinTransformer
from models.SwinSegmentation.uper_head import UPerHead
from models.SwinSegmentation.mmseg.ops import resize
norm_cfg = dict(type='BN', requires_grad=True)


class SwinSegmentationBase(nn.Module):
    def __init__(self, num_classes, shape):
        super(SwinSegmentationBase, self).__init__()
        self.num_classes = num_classes
        self.shape = shape
        self.back_bone = SwinTransformer(
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=False
        )
        self.decoder = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=self.num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        )

    def forward(self, x):
        x = self.back_bone(x)
        x = self.decoder(x)
        x = resize(
            input=x,
            size=self.shape,
            mode='bilinear',
            align_corners=False)
        return x


class SwinSegmentationLarge(nn.Module):
    def __init__(self, num_classes, shape):
        super(SwinSegmentationLarge, self).__init__()
        self.num_classes = num_classes
        self.shape = shape
        self.back_bone = SwinTransformer(
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            use_checkpoint=False
        )
        self.decoder = UPerHead(
            in_channels=[192, 384, 768, 1536],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=self.num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        )

    def forward(self, x):
        x = self.back_bone(x)
        x = self.decoder(x)
        x = resize(
            input=x,
            size=self.shape,
            mode='bilinear',
            align_corners=False)
        return x


def test():
    x = torch.rand(3, 3, 128, 128).cuda()
    model = SwinSegmentationBase(1, [128, 128])
    model = nn.DataParallel(model).cuda()
    pred = model(x)
    print(x.shape)
    print(pred.shape)


if __name__ == '__main__':
    test()
