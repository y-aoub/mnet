from data.data_utils import DataProcessing, DataPaths
import argparse


parser = argparse.ArgumentParser(description='Launch Preprocessing')

parser.add_argument('--no_polar_transform', action='store_true',
                    help='Perform polar transformation of data')

parser.add_argument('--n_data', default=650, type=int, 
                    help='Number of data elements to process')

args = parser.parse_args()

n_data = args.n_data

data_paths = DataPaths()
data_processing = DataProcessing()

raw_images_abs_path = data_paths.RAW_IMAGES_ABS_PATH
raw_masks_abs_path = data_paths.RAW_MASKS_ABS_PATH
processed_images_abs_path = data_paths.PROCESSED_IMAGES_ABS_PATH
processed_masks_abs_path = data_paths.PROCESSED_MASKS_ABS_PATH

raw_masks_files_abs_path = data_processing.list_data_files_abs_path(raw_masks_abs_path)[:n_data]
raw_images_files_abs_path = data_processing.list_data_files_abs_path(raw_images_abs_path)[:n_data]

raw_images = data_processing.read_all_raw_images(raw_images_files_abs_path)
raw_masks = data_processing.read_all_raw_masks(raw_masks_files_abs_path)

raw_data = data_processing.merge_raw_data(raw_images, raw_masks)

data = data_processing.resize_all_masks(raw_data)

if not args.no_polar_transform:
    data = data_processing.polar_transformation_all_data(data)

resized_data = data_processing.resize_all_data(data)

data_processing.create_processed_data_dir()

data_processing.save_data(resized_data)
