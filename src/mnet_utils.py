import torch
from tqdm import tqdm
import monai.transforms as transforms
from metrics import Metrics
import logging
import torchvision.transforms.functional as F


logging.basicConfig(format='%(message)s', level=logging.INFO)

class Transforms:
    def __init__(self, threshold=0.5, values=[0, 1, 2]):
        
        self.threshold = threshold
        self.threshold = values
        # in order 0 (background), 1, 2
        self.as_discrete = transforms.AsDiscrete(to_onehot=self.num_classes)
        
    def thresholding(self, tensor):
        return torch.where(tensor >= self.threshold, 1, 0)

    def one_hot_encoding(self, tensor):
        binary_masks = [torch.eq(tensor, value).float() for value in self.values]
        one_hot_tensor = torch.stack(binary_masks, dim=1).squeeze(2)
        return one_hot_tensor[:, 1:, :, :]

class Trainer:
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, device, weights_path):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.transforms = Transforms()
        self.metrics = Metrics()
        self.best_metric = -1
        self.weights_path = weights_path
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {"mean_dice_1": [], "mean_dice_2": [], "specificity": [], "sensitivity": [], "combined_metric": []}
        self.val_metrics = {"mean_dice_1": [], "mean_dice_2": [], "specificity": [], "sensitivity": [], "combined_metric": []}

    def compute_metrics_and_store(self, preds, target, mode):
        mean_dice_score, specificity, sensitivity, combined_metric = transforms.compute_metrics(preds, target)
        
        metrics_dict = {
            "mean_dice_1": mean_dice_score[0].item(),
            "mean_dice_2": mean_dice_score[1].item(),
            "specificity": specificity.item(),
            "sensitivity": sensitivity.item(),
            "combined_metric": combined_metric.item()
        }

        metrics_dict_list = {"train": self.train_metrics, "val": self.val_metrics}
        
        for key in metrics_dict:
            metrics_dict_list[mode][key].append(metrics_dict[key])

        return metrics_dict
    
    def train_one_batch(self, batch):
        images, masks = batch
        masks = self.transforms.one_hot_encoding(masks)
        images, masks = images.to(self.device), masks.to(self.device)

        self.optimizer.zero_grad()
        conv_outputs, final_output = self.model(images)[:-1], self.model(images)[-1]
        loss_value = self.loss(conv_outputs, masks)
        loss_value.backward()
        self.optimizer.step()

        final_output = self.transforms.thresholding(final_output)
        metrics_dict = self.compute_metrics_and_store(final_output, masks, mode="train")

        return loss_value.item(), metrics_dict

    def validate_one_batch(self, batch):
        images, masks = batch
        masks = self.transforms.one_hot_encoding(masks)
        images, masks = images.to(self.device), masks.to(self.device)

        self.model.eval()
        with torch.no_grad():
            conv_outputs, final_output = self.model(images)[:-1], self.model(images)[-1]
            val_loss = self.loss(conv_outputs, masks)

        final_output = self.transforms.thresholding(final_output)
        metrics_dict = self.compute_metrics_and_store(final_output, masks, mode="val")

        return val_loss.item(), metrics_dict

    def train_one_epoch(self, dataloader, desc):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, unit="batch", desc=desc):
            loss_value, metrics_dict = self.train_one_batch(batch)
            total_loss += loss_value
            
        average_loss = total_loss / len(dataloader)
        self.train_losses.append(average_loss)
        return average_loss, metrics_dict

    def validate_one_epoch(self, dataloader, desc):
        self.model.eval()
        total_val_loss = 0.0
        for batch in tqdm(dataloader, unit="batch", desc=desc):
            val_loss, metrics_dict = self.validate_one_batch(batch)
            total_val_loss += val_loss

        average_val_loss = total_val_loss / len(dataloader)
        self.val_losses.append(average_val_loss)
        if metrics_dict > self.best_metric:
                self.best_metric = metrics_dict["mean_dice_1"]
                torch.save(self.model.state_dict(), self.weights_path)
        return average_val_loss, metrics_dict
    
class Tester:
    def __init__(self, model, loss, test_dataloader, device):
        self.model = model
        self.loss = loss
        self.test_dataloader = test_dataloader
        self.device = device
        self.metrics = Metrics()
        self.transforms = Transforms()
        self.test_losses = []
        self.test_metrics = {"mean_dice_1": [], "mean_dice_2": [], "specificity": [], "sensitivity": [], "combined_metric": []}

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)

    def compute_metrics_and_store(self, preds, target):
        mean_dice_score, specificity, sensitivity, combined_metric = self.metrics.compute_metrics(preds, target)
        metrics_dict = {
            "mean_dice_1": mean_dice_score[0].item(),
            "mean_dice_2": mean_dice_score[1].item(),
            "specificity": specificity.item(),
            "sensitivity": sensitivity.item(),
            "combined_metric": combined_metric.item()
            }
        return metrics_dict
        
    def test_one_batch(self, batch):
        images, masks = batch
        masks = self.transforms.one_hot_encoding(masks)
        images, masks = images.to(self.device), masks.to(self.device)

        with torch.no_grad():
            conv_outputs, final_output = self.model(images)[:-1], self.model(images)[-1]
            test_loss = self.loss(conv_outputs, masks)

        final_output = self.transforms.thresholding(final_output)
        metrics_dict = self.compute_metrics_and_store(final_output, masks)

        self.test_losses.append(test_loss.item())

        return test_loss.item(), metrics_dict

    def test(self):
        self.model.eval()
        total_test_loss = 0.0
        for batch in tqdm(self.test_dataloader, unit="batch", desc="Testing"):
            test_loss, metrics_dict = self.test_one_batch(batch)
            total_test_loss += test_loss

        average_test_loss = total_test_loss / len(self.test_dataloader)
        return average_test_loss, metrics_dict

        
if __name__ == "__main__":
    
    ######## TEST ##########
    pass