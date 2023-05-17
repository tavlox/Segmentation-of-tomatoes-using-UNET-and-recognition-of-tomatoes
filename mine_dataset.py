import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np



class CustomDataset(Dataset):
    def __init__(self, path_original, image_paths, target_paths, transforms_images = None, transforms_masks = None):   

         self.image_paths = image_paths
         self.target_paths = target_paths
         self.transforms_images = transforms_images
         self.transforms_masks = transforms_masks
         #self.transforms = transforms.ToTensor()
         self.path_original = path_original
         self.image_names = sorted(os.listdir(path_original))
         

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        t_image = self.transforms_images(image)
        t_mask = self.transforms_masks(mask)
        image_name_1 = self.image_names[index]
        
        
        return t_image, t_mask, image_name_1

    def __len__(self):  # return count of sample we have

        return len(self.image_names)