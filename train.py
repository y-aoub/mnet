import argparse
from data.data_utils import DataLoadersManager
from src.mnet_utils import Trainer, Tester, Plots, Transforms
from data.data_utils import DataPaths
from src.loss import MNetLoss
from src.mnet import MNet
from torch import optim
import logging
import datetime

CURRENT_DATETIME = datetime.datetime.now()
DATE_TIME_STR = CURRENT_DATETIME.strftime("%m_%d_%H_%M")

default_weights_path = f"./models/weights_{DATE_TIME_STR}.pth"

logging.basicConfig(format='%(message)s', level=logging.INFO)

data_paths = DataPaths()
processed_images_abs_path = data_paths.PROCESSED_IMAGES_ABS_PATH
processed_masks_abs_path = data_paths.PROCESSED_MASKS_ABS_PATH

parser = argparse.ArgumentParser(description="Training script for MNet")

parser.add_argument("--image_dir", type=str, default=processed_images_abs_path, help="Path to the directory containing image data.")
parser.add_argument("--mask_dir", type=str, default=processed_masks_abs_path, help="Path to the directory containing mask data.")
parser.add_argument("--weights_path", type=str, default=default_weights_path, help="Path to the directory containing mask data.")
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size for validation.")
parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for testing.")
parser.add_argument("--train_size", type=int, default=325, help="Size of the training set.")
parser.add_argument("--val_size", type=int, default=75, help="Size of the validation set.")
parser.add_argument("--test_size", type=int, default=250, help="Size of the test set.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer.")
parser.add_argument("--include_background", action="store_true", help="Include background in the loss computation.")
parser.add_argument("--to_onehot_y", action="store_true", help="Convert target masks to one-hot encoding.")
parser.add_argument("--reduction", type=str, default='mean', help="Reduction method for the loss.")
parser.add_argument("--device", type=str, default='cpu', help="Specify device ('cuda' or 'cpu")
parser.add_argument("--test_only", action="store_true", help="Run only the test loop.")

args = parser.parse_args()

image_dir = args.image_dir
mask_dir = args.mask_dir
lr = args.learning_rate
momentum = args.momentum
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
test_batch_size = args.test_batch_size
epochs = args.epochs
include_background = args.include_background
to_onehot_y = args.to_onehot_y
reduction = args.reduction
test_only = args.test_only
train_size = args.train_size 
val_size = args.val_size 
test_size = args.test_size
device = args.device
weights_path = args.weights_path

dataloaders_manager = DataLoadersManager(
    image_dir=image_dir,
    mask_dir=mask_dir,
    train_batch_size=train_batch_size,
    val_batch_size=val_batch_size,
    test_batch_size=test_batch_size,
    train_size=train_size,
    val_size=val_size,
    test_size=test_size
)

dataloaders = dataloaders_manager.get_dataloaders()
train_dataloader = dataloaders['train']
val_dataloader = dataloaders['val']
test_dataloader = dataloaders['test']

model = MNet().to(device)
loss = MNetLoss(include_background, to_onehot_y, reduction).to(device)
optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

if __name__ == "__main__":
    
    if test_only:
        tester = Tester(model, loss, test_dataloader, device)
        tester.load_model(weights_path)
        average_test_loss, metrics_dict = tester.test()
        logging.info(f"Test, Average Loss: {average_test_loss}")
        logging.info(f"Test Metrics: {metrics_dict}")
    else:
        trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, device, weights_path)

        for epoch in range(epochs):
            # Training loop
            average_loss, train_metrics_dict = trainer.train_one_epoch(train_dataloader, f"Epoch {epoch + 1} - Training")
            logging.info(f"Epoch {epoch + 1} - Training, Average Loss: {average_loss}")
            logging.info(f"Epoch {epoch + 1} - Training Metrics: {train_metrics_dict}")

            # Validation loop
            average_val_loss, val_metrics_dict = trainer.validate_one_epoch(val_dataloader, f"Epoch {epoch + 1} - Validation")
            logging.info(f"Epoch {epoch + 1} - Validation, Average Loss: {average_val_loss}")
            logging.info(f"Epoch {epoch + 1} - Validation Metrics: {val_metrics_dict}")

        # Test loop
        tester = Tester(model, loss, test_dataloader, device)
        tester.load_model(weights_path)
        average_test_loss, metrics_dict = tester.test()
        logging.info(f"Test, Average Loss: {average_test_loss}")
        logging.info(f"Test Metrics: {metrics_dict}")