import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import logging


class DiceCeLoss_multiclass(nn.Module):
    def __init__(self):
        super(DiceCeLoss_multiclass, self).__init__()
        self.ce_loss = nn.BCELoss()

    def forward(self, pred, label, label1):
        assert pred[0][0].shape == label[0][0].shape, f"predict & target shape do not match: {pred.shape} vs {label.shape}"
        assert pred[0][1].shape == label1[0][0].shape, f"predict & target1 shape do not match: {pred.shape} vs {label1.shape}"

        ce_loss = self.ce_loss(pred[0][0], label[0][0])
        ce_loss1 = self.ce_loss(pred[0][1], label1[0][0])

        pred0 = pred[0][0].reshape(-1)
        label0 = label[0][0].reshape(-1)
        intersection0 = torch.sum(pred0 * label0)
        dice0 =  (2. * intersection0 + 1e-8) / (torch.sum(pred0) + torch.sum(label0) + 1e-8)

        pred1 = pred[0][1].reshape(-1)
        label1 = label1[0][0].reshape(-1)
        intersection1 = torch.sum(pred1 * label1)
        dice1 =  (2. * intersection1 + 1e-8) / (torch.sum(pred1) + torch.sum(label1) + 1e-8)

        ce_loss2 = 0.3 * ce_loss + 0.7 * ce_loss1
        dice2 = 0.3 * dice0 + 0.7 * dice1
        # logging.info("Dice: %s", dice_i)
        # logging.info("CE total loss: %s", ce_loss)
        dice2 = dice2.clamp(0, 1)
        dice2[dice2 < 1e-8] = 1e-8
        return ce_loss2 - 0.5 * torch.log(dice2), ce_loss, ce_loss1

class DiceCeLoss(nn.Module):
    def __init__(self):
        super(DiceCeLoss, self).__init__()
        self.ce_loss = nn.BCELoss()

    def forward(self, preds, preds1, labels, labels1):
        assert preds.shape == labels.shape, f"predict & target shape do not match: {preds.shape} vs {labels.shape}"

        labels = labels + labels1

        labels = torch.where(labels >= 0.999, torch.tensor(1.0, device=labels.device), torch.tensor(0.0, device=labels.device))
        labels1 = torch.where(labels1 >= 0.999, torch.tensor(1.0, device=labels1.device), torch.tensor(0.0, device=labels1.device))

        ce_loss = self.ce_loss(preds, labels)
        ce_loss1 = self.ce_loss(preds1, labels1)

        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        intersection = torch.sum(preds * labels)
        dice_i =  (2. * intersection + 1e-8) / (torch.sum(preds) + torch.sum(labels) + 1e-8)

        preds1 = preds1.reshape(-1)
        labels1 = labels1.reshape(-1)
        intersection1 = torch.sum(preds1 * labels1)
        dice_i1 =  (2. * intersection1 + 1e-8) / (torch.sum(preds1) + torch.sum(labels1) + 1e-8)

        dice_i2 = dice_i * 0.3 + dice_i1 * 0.7
        ce_loss2 = ce_loss * 0.3 + ce_loss1 * 0.7
        # logging.info("Dice: %s", dice_i)
        # logging.info("CE total loss: %s", ce_loss)
        loss0 = ce_loss - 0.5 * torch.log(dice_i)
        loss1 = ce_loss1 - 0.5 * torch.log(dice_i1)
        return ce_loss2 - 0.5 * torch.log(dice_i2), loss0, loss1

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2): 
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7  

        loss_y1 = -1 * self.alpha * \
            torch.pow((1 - preds), self.gamma) * \
            torch.log(preds + eps) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds,
                                                    self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)

class MultiFocalLoss(nn.Module):
    def __init__(self):
        super(MultiFocalLoss, self).__init__()
        self.binary_focal_loss = BinaryFocalLoss()
        
    def forward(self, pred, label, label1):
        focal_loss0 = self.binary_focal_loss(pred, label)
        focal_loss1 = self.binary_focal_loss(pred, label1)

        total_focal = 0.5 * focal_loss0 + 0.5 * focal_loss1

        pred0 = pred[0][0].reshape(-1)
        label0 = label[0][0].reshape(-1)
        intersection0 = torch.sum(pred0 * label0)
        dice0 =  (2. * intersection0 + 1e-8) / (torch.sum(pred0) + torch.sum(label0) + 1e-8)

        pred1 = pred[0][1].reshape(-1)
        label1 = label1[0][0].reshape(-1)
        intersection1 = torch.sum(pred1 * label1)
        dice1 =  (2. * intersection1 + 1e-8) / (torch.sum(pred1) + torch.sum(label1) + 1e-8)

        dice2 = 0.3 * dice0 + 0.7 * dice1
        dice2 = dice2.clamp(0, 1)
        dice2[dice2 < 1e-8] = 1e-8
        
        return total_focal - 0.5 * torch.log(dice2), focal_loss0, focal_loss1



def uncertainty_to_weigh_losses(loss_list):
    loss_n = len(loss_list)
    uncertainty_weight = [
        nn.Parameter(torch.tensor([1 / loss_n]), requires_grad=True) for _ in range(loss_n)
    ]

    final_loss = []
    for i in range(loss_n):
        final_loss.append(loss_list[i] / (2 * uncertainty_weight[i]**2) + torch.log(uncertainty_weight[i]))

    return sum(final_loss)

    
