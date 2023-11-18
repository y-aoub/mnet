# Preprocessing Script

The `preprocess.py` script is designed to preprocess a dataset using the `DataProcessing` class from the `data.data_utils` module. The preprocessing includes reading raw images and masks, resizing them, and optionally performing a polar transformation.

## Usage

Before running the script, make sure you have the necessary data directories: `raw_images` and `raw_masks` in the parent `data` directory. Find the link to download data [here](https://drive.google.com/file/d/1svVhcxeeEFaEU-MlQSYH-madg6aYx2h6/view?usp=sharing): 

### Command-line Arguments

- `--no_polar_transform`: If specified, the script will skip the polar transformation step and keep the data in the Cartesian coordinates system.
- `--n_data`: Number of data elements to process (default is 650, which is the total number of data elements in the ORIGA dataset).

### Running the Script

```bash
python preprocess.py [--no_polar_transform] [--n_data N]
