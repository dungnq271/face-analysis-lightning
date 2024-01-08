import os.path as osp
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from .transforms import get_img_trans


class FaceDataset(Dataset):
    """Fashion Color dataset."""

    def __init__(self,
                 root_dir,
                 image_list,
                 mean,
                 std,
                 image_size,
                 crop_size,
                 mode="train",
                 transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(image_list, delimiter=" ", header=None)
        self.data = np.array(self.data)
        self.root_dir = root_dir
        if transform:
            self.transform = get_img_trans(mode,
                                           image_size=image_size,
                                           crop_size=crop_size,
                                           mean=mean,
                                           std=std)
        self.age_classes = 6
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = osp.join(self.root_dir, self.data[idx][0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.data[1:]
        age = label[2]
        if self.transform:
            image = self.transform(image=image)["image"]
        label = self._transform_ages_to_one_hot_ordinal(label,
                                                        age,
                                                        self.age_classes
                                                        )
        return image, label

    def _transform_ages_to_one_hot_ordinal(self, label, age, age_classes):
        new_target = np.zeros(shape=age_classes)
        new_target[:age] = 1
        label = label[:2] + new_target + label[3:]
        return label
