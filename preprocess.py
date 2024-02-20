from data.data_utils import DataProcessing, DataPaths, DataUtils
from tqdm import tqdm
import logging
import argparse


parser = argparse.ArgumentParser(description='Launch Preprocessing')

logging.basicConfig(format='%(message)s', level=logging.INFO)

parser.add_argument('--no_polar_transform', action='store_true',
                    help='Perform polar transformation of data')

parser.add_argument('--n_data', default=650, type=int, 
                    help='Number of data elements to process')

args = parser.parse_args()

no_polar_transform = args.no_polar_transform
n_data = args.n_data

data_paths = DataPaths()
data_processing = DataProcessing()
data_utils = DataUtils()

raw_images_abs_path = data_paths.RAW_IMAGES_ABS_PATH
raw_masks_abs_path = data_paths.RAW_MASKS_ABS_PATH
processed_images_abs_path = data_paths.PROCESSED_IMAGES_ABS_PATH
processed_masks_abs_path = data_paths.PROCESSED_MASKS_ABS_PATH

raw_masks_files_abs_path = data_utils.list_data_files_abs_path(raw_masks_abs_path)[:n_data]
raw_images_files_abs_path = data_utils.list_data_files_abs_path(raw_images_abs_path)[:n_data]

data_utils.create_processed_data_dir()

raw_data_files_abs_path = list(zip(raw_images_files_abs_path, raw_masks_files_abs_path))

if __name__ == "__main__":
    logging.info("\nProcessing data")

    for raw_image_file_path, raw_mask_file_path in tqdm(raw_data_files_abs_path):
        
        basename = data_utils.get_file_basename(raw_image_file_path)
        
        image = data_utils.read_jpg_file(raw_image_file_path)
        mask = data_utils.read_mat_file(raw_mask_file_path)
        
        image, mask = data_processing.resize_mask(image, mask)
        
        if not no_polar_transform:
            image, mask = data_processing.polar_transform_data_tuple(image, mask)
            
        image, mask = data_processing.resize_data_tuple(image, mask)
        
        data_utils.save_data_tuple(image, mask, basename)
        
    logging.info("Preprocessed data successfully saved")

    