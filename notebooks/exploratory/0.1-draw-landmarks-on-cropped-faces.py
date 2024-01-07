# %%
import os
import os.path as osp
import random
import json

import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from src.reproducible_code.tools import plot, image_utils

# %%
data_dir = "../../data/interim"
img_dir = osp.join(data_dir, "cropped_faces")
img_fns = os.listdir(img_dir)
len(img_fns)

# %%
face_landmarks = json.load(open(osp.join(data_dir, "face_landmarks.json")))
face_landmarks["1.jpg"]

# %%
# Radius of circle 
sample_img_fns = random.sample(img_fns, 16)
face_imgs = []

radius = 1

# Blue color in BGR 
color = (255, 0, 0) 

# Line thickness of 2 px 
thickness = 2

for fn in sample_img_fns:
    fp = osp.join(img_dir, fn)
    face_img = image_utils.load_image(fp)
    h, w, _ = face_img.shape
    landmarks = face_landmarks[fn]

    for ldm in landmarks:
        x = ldm[0] * w
        y = ldm[1] * h
        # Using cv2.circle() method 
        # Draw a circle with blue line borders of thickness of 2 px 
        face_img = cv2.circle(face_img, (int(x), int(y)), radius, color, thickness) 
    face_imgs.append(face_img)

# %%
plot.display_multiple_images(face_imgs)

# %%
