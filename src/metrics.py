import torch
from monai.metrics import DiceMetric
from mnet import MNet
from mnet_utils import Transforms

class Metrics:
    def __init__(self):
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

    def compute_mean_dice(self, preds, target):
        dice_score = self.dice_metric(preds, target)
        mean_dice_score_0, mean_dice_score_1 = torch.mean(dice_score[:, 0]), torch.mean(dice_score[:, 1])
        mean_dice_score = torch.stack([mean_dice_score_0, mean_dice_score_1])
        return mean_dice_score

    def compute_specificity(self, preds, target):
        true_negative = torch.sum((1 - preds) * (1 - target))
        actual_negative = torch.sum(1 - target)
        specificity = true_negative / actual_negative
        return specificity

    def compute_sensitivity(self, preds, target):
        true_positive = torch.sum(preds * target)
        actual_positive = torch.sum(target)
        sensitivity = true_positive / actual_positive
        return sensitivity
    
    def compute_metrics(self, preds, target):
        mean_dice_score = self.compute_mean_dice(preds, target)
        specificity = self.compute_specificity(preds, target)
        sensitivity = self.compute_sensitivity(preds, target)
        combined_metric = 0.5 * (specificity + sensitivity)
        return mean_dice_score, specificity, sensitivity, combined_metric

if __name__ == "__main__":
    metrics = Metrics()
    model = MNet()
    transforms = Transforms()

    # target [1, 1, 400, 400]
    target = torch.randint(0, 3, size=(2, 1, 400, 400))
    print(f"TARGET SHAPE : {target.shape}")
    
    # target_one_hot [1, 3, 400, 400]
    post_target = transforms.one_hot_encoding(target)
    print(f"TRAGET_ONE_HOT SHAPE : {post_target.shape}")
    
    # input image [1, 3, 400, 400]
    images = torch.randn([2, 3, 400, 400])
    print(f"IMAGE SHAPE : {images.shape}")
    
    # model output [1, 2, 400, 400]
    _, pred = model(images)[:-1], model(images)[-1]
    print(f"PRED SHAPE : {pred.shape}")
    
    # model output thresholded [1, 2, 400, 400]
    post_pred = transforms.thresholding(pred)
    print(f"PRED_THRESHOLDED SHAPE : {post_pred.shape}")
    

    
    
    
    # post_target_0 = post_target[:, 0, :, :]
    # post_target_1 = post_target[:, 1, :, :]
    # post_target_2 = post_target[:, 2, :, :]
    
    # post_pred_0 = post_pred[:, 0, :, :]
    # post_pred_1 = post_pred[:, 1, :, :]
    # post_pred_2 = post_pred[:, 2, :, :]
    
    
    mean_dice_score, specificity, sensitivity, combined_metric = metrics.compute_metrics(post_pred, post_target)

    print("Dice Score:", mean_dice_score)
    print("Specificity:", specificity)
    print("Sensitivity:", sensitivity)
    print("Combined Metric:", combined_metric)
