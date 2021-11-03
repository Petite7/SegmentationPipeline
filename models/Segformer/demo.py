from mix_transformer import *
from segformer_head import SegFormerHead
from wrappers import resize
import torch 
import torch.nn as nn

norm_cfg = dict(type='SyncBN', requires_grad=True)


class SegNet(nn.Module):
    def __init__(self, num, shape):
        super(SegNet, self).__init__()
        self.shape = shape
        self.backbone = mit_b2()
        self.decoder = SegFormerHead(in_channels=[64, 128, 320, 512],
                        in_index=[0, 1, 2, 3],
                        feature_strides=[4, 8, 16, 32],
                        channels=128,
                        dropout_ratio=0.1,
                        num_classes=num,
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


if __name__ == '__main__':
    hw = 508
    t1 = torch.rand(2, 3, hw, hw)
    net = SegNet(2, [hw, hw])
    net = nn.DataParallel(net).cuda()
    t2 = net(t1)
    print('input:', t1.shape)
    print('output:', t2.shape)
