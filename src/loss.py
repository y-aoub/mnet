from monai.losses import DiceLoss
import torch.nn as nn
import torch

class MNetLoss(nn.Module):
    def __init__(self, include_background=False, to_onehot_y=False, reduction='mean'):
        super(MNetLoss, self).__init__()
        self.dice_loss = DiceLoss(include_background=include_background, to_onehot_y=to_onehot_y, reduction=reduction)
            
    def forward(self, preds, target):
        losses = [self.dice_loss(pred, target) for pred in preds]
        mean_loss = torch.mean(torch.stack(losses))
        return mean_loss
    
if __name__ == "__main__":
    
    ######## TEST ##########
    pass