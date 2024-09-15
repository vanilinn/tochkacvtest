import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import FCN_ResNet50_Weights


class WeightedIOULoss(nn.Module):
    def __init__(self, smooth=1e-6, min_area_weight=5, area_threshold=0.03):
        super(WeightedIOULoss, self).__init__()
        self.smooth = smooth
        self.min_area_weight = min_area_weight
        self.area_threshold = area_threshold

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        targets = targets.float()
        intersection = (outputs * targets).sum((2, 3))
        union = (outputs + targets).sum((2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        mask_area = targets.sum((2, 3)) / (targets.size(2) * targets.size(3))
        small_object_weight = (mask_area < self.area_threshold).float() * self.min_area_weight
        weighted_iou = iou * small_object_weight + iou * (1 - small_object_weight)

        return 1 - weighted_iou.mean()


class IOULoss(nn.Module):
    def __init__(self, smooth=1e-6, weights=None):
        super(IOULoss, self).__init__()
        self.smooth = smooth
        self.weights = weights

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        targets = targets.float()
        intersection = (outputs * targets).sum((2, 3))
        union = (outputs + targets).sum((2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        if self.weights is not None:
            assert len(self.weights) == iou.size(1), "Weights length must match the number of classes"
            iou = iou * torch.tensor(self.weights, device=iou.device)
        return 1 - iou.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * targets).sum((2, 3))
        dice = (2. * intersection + self.smooth) / (outputs.sum((2, 3)) + targets.sum((2, 3)) + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, eps=1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() + self.eps


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        true_pos = (outputs * targets).sum((2, 3))
        false_neg = (targets * (1 - outputs)).sum((2, 3))
        false_pos = ((1 - targets) * outputs).sum((2, 3))
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky.mean()


class UNet(nn.Module):
    def __init__(self, encoder_name="efficientnet_b1", pretrained=True, num_classes=1, dropout_prob=0.5):
        super(UNet, self).__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = [512, 256, 128, 64]  # Каналы на декодирующих уровнях
        self.decoder5 = self._decoder_block(encoder_channels[4], decoder_channels[0], dropout_prob)
        self.decoder4 = self._decoder_block(encoder_channels[3] + decoder_channels[0], decoder_channels[1],
                                            dropout_prob)
        self.decoder3 = self._decoder_block(encoder_channels[2] + decoder_channels[1], decoder_channels[2],
                                            dropout_prob)
        self.decoder2 = self._decoder_block(encoder_channels[1] + decoder_channels[2], decoder_channels[3],
                                            dropout_prob)
        self.decoder1 = self._decoder_block(encoder_channels[0] + decoder_channels[3], decoder_channels[3],
                                            dropout_prob)
        self.output_conv = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        enc1, enc2, enc3, enc4, enc5 = self.encoder(x)
        dec5 = self.decoder5(enc5)
        dec4 = self.decoder4(
            torch.cat([F.interpolate(dec5, size=enc4.shape[2:], mode='bilinear', align_corners=False), enc4], dim=1))
        dec3 = self.decoder3(
            torch.cat([F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False), enc3], dim=1))
        dec2 = self.decoder2(
            torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False), enc2], dim=1))
        dec1 = self.decoder1(
            torch.cat([F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False), enc1], dim=1))
        output = self.output_conv(dec1)
        output = F.interpolate(output, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output


def get_model(pretrained=True, dropout_prob=0.5):
    model = UNet(encoder_name="efficientnet_b1", pretrained=pretrained, num_classes=1, dropout_prob=dropout_prob)
    return model


# Комбинация BCE + Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss(weight=weight)

    def dice_loss(self, pred, target, smooth=1.):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) /
                     (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return bce_loss + dice_loss
