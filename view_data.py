from data.data_utils import DataUtils, DataPaths
import argparse
import os

parser =argparse.ArgumentParser(description='Vizualise Data Element')

parser.add_argument('--data_element', type=str, 
                    help="Specify the file name of the data element to vizualise (001 to 650)")
parser.add_argument('--raw', action='store_true', 
                    help="Add --raw to see raw data else viszalise processed data")


args = parser.parse_args()

data_element = args.data_element
raw = args.raw

data_paths = DataPaths()
data_utils = DataUtils()

processed_images_abs_path = data_paths.PROCESSED_IMAGES_ABS_PATH
processed_masks_abs_path = data_paths.PROCESSED_MASKS_ABS_PATH
raw_images_abs_path = data_paths.RAW_IMAGES_ABS_PATH
raw_masks_abs_path = data_paths.RAW_MASKS_ABS_PATH

if raw:
    data_element_image, data_element_mask = f"{data_element}.jpg", f"{data_element}.mat"
    data_element_image_abs_path = os.path.join(raw_images_abs_path, data_element_image)
    data_element_mask_abs_path = os.path.join(raw_masks_abs_path, data_element_mask)
    image = data_utils.read_jpg_file(data_element_image_abs_path)
    mask = data_utils.read_mat_file(data_element_mask_abs_path)
    
else:
    data_element_image, data_element_mask = f"{data_element}.npy", f"{data_element}.npy"
    data_element_image_abs_path = os.path.join(processed_images_abs_path, data_element_image)
    data_element_mask_abs_path = os.path.join(processed_masks_abs_path, data_element_mask)
    image = data_utils.read_npy_file(data_element_image_abs_path)
    mask = data_utils.read_npy_file(data_element_mask_abs_path)
    
    
if __name__ == "__main__":
    data_utils.plot_data_element(image, mask)