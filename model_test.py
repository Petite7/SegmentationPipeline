import torch
import torch.nn as nn
from models.Unet import ClassicUnet
from models.Segformer.SegFormer import SegFormerNet
from torchvision import models


def test():
    # x : tensor - [batch_size, channel, w, h]
    x = torch.randn((8, 3, 64, 64))

    # model = ClassicUnet(in_channels=1, out_channels=1)
    model = SegFormerNet(1, [64, 64])
    # model = models.segmentation.fcn_resnet101(False)
    # model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))

    model = nn.DataParallel(model).cuda()
    pred = model(x)
    print(x.shape)
    print(pred.shape)
    # print(pred['out'].shape)
    # assert pred.shape == x.shape


if __name__ == '__main__':
    test()
