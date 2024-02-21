
# Setup

## Downloading Data

Before running the script, download the ORIGA dataset and unzip it using the following commands:

```bash
wget -O data/data.zip https://drive.google.com/uc\?export\=download\&id\=1svVhcxeeEFaEU-MlQSYH-madg6aYx2h6
unzip -o data/data.zip -d data/
```
## Environment Setup

Create and activate virtual environment (optional but recommended):

```bash
python -m venv .env
source .env/bin/activate
```

Install Python packages from the requirements.txt file:

```bash
pip install -r requirements.txt
```
# Run

## Preprocessing

`preprocess.py` is used to process the the raw images and their corresponding raw masks. As mentionned in the artice.

### Arguments
- `--no_polar_transform`: If specified, the script will skip the polar transformation step, preserving the data in the Cartesian coordinate system.
- `--n_data`: Number of data elements (image, and mask) to process, (default is 650, the total number in the ORIGA dataset).

### Running the Script

```bash
python preprocess.py [--no_polar_transform] [--n_data]
```

## Training

`train.py` is used to sequentially train, validate, and test the model on the training, validation, and testing datasets after partitioning the entire dataset.

### Running the Script

```bash
python train.py [--image_dir] [--mask_dir][--weights_path][--train_batch_size][--val_batch_size] [--test_batch_size] [--train_size][--val_size] [--test_size] [--epochs] [--learning_rate][--momentum][--include_background] [--to_onehot_y] [--reduction] [--device] [--test_only]
  ```

### Arguments
- `--image_dir`: Path to the directory containing image data (default is `./data/processed_images/`).
- `--mask_dir`: Path to the directory containing mask data (default is `./data/processed_masks/`).
- `--weights_path`: Path to the directory containing initial weights for the model.
- `--train_batch_size`: Batch size for training (default is 4).
- `--val_batch_size`: Batch size for validation (default is 4).
- `--test_batch_size`: Batch size for testing (default is 4).
- `--train_size`: Size of the training set (default is 325).
- `--val_size`: Size of the validation set (default is 75).
- `--test_size`: Size of the test set (default is 250).
- `--epochs`: Number of epochs for training (default is 10).
- `--learning_rate`: Learning rate for the optimizer (0.0001).
- `--momentum`: Momentum for the optimizer (default is 0.9).
- `--include_background`: Include background in the loss computation (default is False; If not specified, background is not included).
- `--to_onehot_y`: Convert target masks to one-hot encoding (defualt is False; if not specified, target masks are not converted to one-hot encoding).
- `--reduction`: Reduction method for the loss (default is 'mean').
- `--device`: Specify device: 'cuda' or 'cpu' (default is 'cpu').
- `--test_only`: Run only the test loop (default is False; if not specified, training and validation loops will be executed in addition to the test loop).

## Testing

`train.py` is used not only for training the model but also for testing it. This is performed after dividing the entire dataset into training, validation, and test datasets. 

### Running the Script

To conduct testing using a pre-trained model with specified weights, execute the `train.py` script as demonstrated in the Training part above. However, ensure to include the `--test_only` and `--weights_path` options in the command.

### Arguments

- `--weights_path`: Path to the pre-trained weights for the model to test (specify the path to the model's pre-trained weights. This allows you to test the model without retraining).
- `--test_only`: Flag to run only the test loop (when this flag is specified, the script will only execute the testing loop, skipping training and validation). 