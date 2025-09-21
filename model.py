# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import EfficientNet_V2_M_Weights

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        target = F.one_hot(target, num_classes=pred.size(-1))
        target = target.float()
        target = (1 - self.smoothing) * target + self.smoothing / pred.size(-1)
        log_pred = F.log_softmax(pred, dim=self.dim)
        loss = nn.KLDivLoss(reduction='batchmean')(log_pred, target)
        return loss


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
     
        self.cnn = torchvision.models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT).cuda()
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.cnn.classifier = nn.Sequential(
            nn.Linear(self.cnn.classifier[1].in_features, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 4),     
        )
        
    def forward(self, img):
        output = self.cnn(img)
        return output