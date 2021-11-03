import torch
import torch.nn as nn


def check_accuracy_binary(loader, model, ttach=False, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pre = torch.sigmoid(model(x))
            pre = (pre > 0.5).float()
            num_correct += (pre == y).sum()
            num_pixels += torch.numel(pre)
            # this dice_score is just for binary
            dice_score += (2 * (pre * y).sum()) / ((pre + y).sum() + 1e-5)

    acc = num_correct/num_pixels*100
    dice = dice_score / len(loader)

    model.train()
    return acc, dice


class DiceLossPlusBECLoss(nn.Module):
    def __init__(self, eps=1e-5, k1=0.3, k2=0.7):
        super(DiceLossPlusBECLoss, self).__init__()
        self.eps = eps
        self.k1 = k1
        self.k2 = k2

    def forward(self, predict, target):
        _bce = nn.BCEWithLogitsLoss()
        _loss1 = _bce(predict, target)

        # dice loss
        activation = nn.Sigmoid()
        predict_sigmoid = activation(predict)

        n = target.size(0)
        pred_flat = predict_sigmoid.view(n, -1)
        tar_flat = target.view(n, -1)

        tp = torch.sum(tar_flat * pred_flat, dim=1)
        fp = torch.sum(pred_flat, dim=1) - tp
        fn = torch.sum(tar_flat, dim=1) - tp
        _losses = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        _loss2 = torch.mean(_losses)

        loss = self.k1*_loss2 + self.k2*_loss1
        return loss



