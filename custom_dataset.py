import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from pathlib import Path 



class CustomDatasetFromImages(Dataset):
    def __init__(self, images_path, downsample=False):
        """
        Args:
            images_path (string): path to folder containing images
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # First column contains the image paths
        self.downsample = downsample
        self.image_arr = [str(e) for e in Path(images_path).rglob("*.jpg")]
        # Calculate len
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        if self.downsample:
            img_as_tensor = img_as_tensor[:,13:-13,13:-13]

        return (img_as_tensor, img_as_tensor)

    def __len__(self):
        return self.data_len



if __name__ == "__main__":

    custom_mnist_from_images =  CustomDatasetFromImages('lfw')

    dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_images,batch_size=10,shuffle=True)
