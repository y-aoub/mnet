import scipy.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import os
import torch
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader, random_split

logging.basicConfig(format='%(message)s', level=logging.INFO)

class DataPaths:
    def __init__(self):
        self.DATA_ABS_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RAW_IMAGES_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "raw_images") 
        self.RAW_MASKS_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "raw_masks")
        self.PROCESSED_IMAGES_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "processed_images")
        self.PROCESSED_MASKS_ABS_PATH = os.path.join(self.DATA_ABS_PATH, "processed_masks")
        
class DataUtils:
    def __init__(self): 
        self.data_paths = DataPaths()
        
    def list_data_files_abs_path(self, abs_path: str) -> list:
        data_files_abs_path = list(map(lambda x: os.path.join(abs_path, x), os.listdir(abs_path)))
        return sorted(data_files_abs_path)
    
    def read_mat_file(self, file_abs_path: str):
        mat_data = scipy.io.loadmat(file_abs_path)
        mask = mat_data['mask']
        return mask
    
    def read_npy_file(self, file_abs_path: str):
        npy_data = np.load(file_abs_path)
        return npy_data
            
    def read_jpg_file(self, file_abs_path):
        image = mpimg.imread(file_abs_path)
        return image
    
    def get_file_basename(self, file_abs_path):
        basename = os.path.splitext(os.path.basename(file_abs_path))[0]
        return basename
            
    def create_processed_data_dir(self):
        if not os.path.exists(self.data_paths.PROCESSED_IMAGES_ABS_PATH):
            os.makedirs(self.data_paths.PROCESSED_IMAGES_ABS_PATH)
        if not os.path.exists(self.data_paths.PROCESSED_MASKS_ABS_PATH):
            os.makedirs(self.data_paths.PROCESSED_MASKS_ABS_PATH)   
            
    def save_data_tuple(self, image, mask, basename):
        saved_image_abs_path = f'{self.data_paths.PROCESSED_IMAGES_ABS_PATH}/{basename}.npy' 
        np.save(saved_image_abs_path, image)
        saved_mask_abs_path = f'{self.data_paths.PROCESSED_MASKS_ABS_PATH}/{basename}.npy'
        np.save(saved_mask_abs_path, mask)
        
    def save_data_tuple(self, image, mask, basename):
        saved_image_abs_path = f'{self.data_paths.PROCESSED_IMAGES_ABS_PATH}/{basename}.npy' 
        np.save(saved_image_abs_path, image)
        saved_mask_abs_path = f'{self.data_paths.PROCESSED_MASKS_ABS_PATH}/{basename}.npy'
        np.save(saved_mask_abs_path, mask)
        
    def plot_data_element(self, image, mask) -> None:
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Image')
        plt.subplot(1, 2, 2), plt.imshow(mask, cmap='gray'), plt.title('Mask')
        plt.show() 
        
class FormatDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        
        data_utils = DataUtils()
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        self.image_files = data_utils.list_data_files_abs_path(image_dir)
        self.mask_files = data_utils.list_data_files_abs_path(mask_dir)
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_path = self.mask_files[index]

        image = np.load(image_path)
        mask = np.load(mask_path)

        image = torch.FloatTensor(image).permute(2, 0, 1)  
        mask = torch.LongTensor(mask).unsqueeze(0)

        return image, mask

class DataLoadersManager:
    def __init__(self, image_dir, mask_dir, train_batch_size, val_batch_size, test_batch_size, train_size, val_size, test_size):
        dataset = FormatDataset(image_dir, mask_dir)
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        
    def get_dataloaders(self):
        return {'train': self.train_dataloader, 'val': self.val_dataloader, 'test': self.test_dataloader}

class DataProcessing:
    def __init__(self, R=400, size=400):
        self.R = R
        self.size = size        
        
    def resize_mask(self, image, mask):
        _, mask_width = mask.shape[:2]
        _, image_width = image.shape[:2]
        start_col = (mask_width - image_width) // 2
        end_col = start_col + image_width
        cropped_mask = mask[:, start_col:end_col]
        return image, cropped_mask
    
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
    
    def polar_transform_data_tuple(self, image, mask) -> None:
        center = self.detect_optic_disc_center(mask)
        polar_image = self.polar_transform_image(image, center=center, R=self.R)
        polar_mask = self.polar_transform_mask(mask, center=center, R=self.R)
        polar_image = cv2.rotate(polar_image, - cv2.ROTATE_90_CLOCKWISE)
        polar_mask = cv2.rotate(polar_mask, - cv2.ROTATE_90_CLOCKWISE)
        return polar_image, polar_mask
    
    def resize_data_tuple(self, image, mask):
        new_size = (self.size, self.size)
        resized_image, resized_mask = cv2.resize(image, new_size), cv2.resize(mask, new_size)
        return resized_image, resized_mask
