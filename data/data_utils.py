import scipy.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import logging

logging.basicConfig(format='%(message)s', level=logging.INFO)

class DataPaths:
    def __init__(self):
        self.DATA_ABS_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RAW_IMAGES_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "raw_images") 
        self.RAW_MASKS_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "raw_masks")
        self.PROCESSED_IMAGES_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "processed_images")
        self.PROCESSED_MASKS_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "processed_masks")
        
class DataProcessing:
    def __init__(self):
        
        self.data_paths = DataPaths()
    
    def list_data_files_abs_path(self, abs_path: str) -> list:
        data_files_abs_path = list(map(lambda x: os.path.join(abs_path, x), os.listdir(abs_path)))
        return sorted(data_files_abs_path)
    
    def read_raw_mask(self, file_abs_path: str):
        mat_data = scipy.io.loadmat(file_abs_path)
        mask = mat_data['mask']
        return mask
    
    def read_raw_image(self, file_abs_path):
        image = mpimg.imread(file_abs_path)
        return image
            
    def create_processed_data_dir(self):
        if not os.path.exists(self.data_paths.PROCESSED_IMAGES_ABS_PATH):
            os.makedirs(self.data_paths.PROCESSED_IMAGES_ABS_PATH)
        if not os.path.exists(self.data_paths.PROCESSED_MASKS_ABS_PATH):
            os.makedirs(self.data_paths.PROCESSED_MASKS_ABS_PATH)
            
    def read_all_raw_images(self, data_files_abs_path):
        logging.info("Reading Images ...")
        raw_images = list(map(lambda x: self.read_raw_image(x), tqdm(data_files_abs_path)))
        return raw_images
    
    def read_all_raw_masks(self, data_files_abs_path: list):
        logging.info("\nReading Masks ...")
        raw_masks = list(map(lambda x: self.read_raw_mask(x), tqdm(data_files_abs_path)))
        return raw_masks
    
    def merge_raw_data(self, raw_images, raw_masks):
        merged_raw_data = list(zip(raw_images, raw_masks))
        return merged_raw_data
    
    def resize_mask(self, image, mask):
        _, mask_width = mask.shape[:2]
        _, image_width = image.shape[:2]
        start_col = (mask_width - image_width) // 2
        end_col = start_col + image_width
        cropped_mask = mask[:, start_col:end_col]
        return image, cropped_mask

    
    def resize_all_masks(self, merged_raw_data):
        logging.info("\nResizing Masks ...")
        merged_data = list(map(lambda x: self.resize_mask(*x), tqdm(merged_raw_data)))
        return merged_data
    
    def detect_optic_disc_center(self, mask):
        mask_copy = np.copy(mask)
        mask_copy[mask_copy == 2] = 1
        contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No ellipse detected")
            return None
        ellipse = cv2.fitEllipse(contours[0])
        center, _, _ = ellipse
        return center
    
    def polar_transform_image(self, image, center, R):
        b, g, r = cv2.split(image)
        polar_b = cv2.linearPolar(b, center, R, cv2.WARP_FILL_OUTLIERS)
        polar_g = cv2.linearPolar(g, center, R, cv2.WARP_FILL_OUTLIERS)
        polar_r = cv2.linearPolar(r, center, R, cv2.WARP_FILL_OUTLIERS)
        polar_image = cv2.merge([polar_b, polar_g, polar_r])
        return polar_image
    
    def polar_transform_mask(self, mask, center, R):
        polar_mask = cv2.linearPolar(mask, center, R, cv2.WARP_FILL_OUTLIERS)
        return polar_mask
    
    def polar_transformation(self, image, mask, R=400) -> None:
        center = self.detect_optic_disc_center(mask)
        polar_image = self.polar_transform_image(image, center=center, R=R)
        polar_mask = self.polar_transform_mask(mask, center=center, R=R)
        polar_image = cv2.rotate(polar_image, - cv2.ROTATE_90_CLOCKWISE)
        polar_mask = cv2.rotate(polar_mask, - cv2.ROTATE_90_CLOCKWISE)
        return polar_image, polar_mask
    
    def polar_transformation_all_data(self, merged_data):
        logging.info("\nPolar Transformation of Data ...")
        polar_data = list(map(lambda x: self.polar_transformation(*x), tqdm(merged_data)))
        return polar_data
    
    def resize_data_element(self, image, mask, size=400):
        new_size = (size, size)
        resized_image, resized_mask = cv2.resize(image, new_size), cv2.resize(mask, new_size)
        return resized_image, resized_mask
    
    def resize_all_data(self, merged_data):
        resized_merged_data = list(map(lambda x: self.resize_data_element(*x), merged_data))
        return resized_merged_data
    
    def plot_data_element(self, image, mask) -> None:
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Image')
        plt.subplot(1, 2, 2), plt.imshow(mask, cmap='gray'), plt.title('Mask')
        plt.show()
        
    def save_data(self, data):
        logging.info("\nSaving Processed Data ...")
        for index in tqdm(range(len(data))):
            image, mask = data[index][0], data[index][1]
            saved_image_abs_path = f'{self.data_paths.PROCESSED_IMAGES_ABS_PATH}/{index+1}.npy' 
            np.save(saved_image_abs_path, image)
            saved_mask_abs_path = f'{self.data_paths.PROCESSED_MASKS_ABS_PATH}/{index+1}.npy'
            np.save(saved_mask_abs_path, mask)
            
