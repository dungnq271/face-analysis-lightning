# %%
import os
import os.path as osp
import json
import random

import pandas as pd
import cv2
from src.reproducible_code.tools import plot, image_utils

# %%
data_dir = "../../data/interim"
img_dir = osp.join(data_dir, "cropped_faces")
label_file = osp.join(data_dir, "labels.csv")
landmarks = json.load(open(osp.join(data_dir, "face_landmarks.json")))
len(landmarks.keys())

# %%
labels_df = pd.read_csv(label_file)
print(len(labels_df))
labels_df.head()

# %%
# Radius of circle 
radius = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
   
# Line thickness of 2 px 
thickness = 2
   
df = labels_df[labels_df["masked"] == "masked"].sample(16)
fns = df.face_file_name.tolist()
sample_fns = random.sample(fns, 16)
face_imgs = []

for fn in sample_fns:
    fp = osp.join(img_dir, fn)
    face_img = image_utils.load_image(fp)

    h, w, _ = face_img.shape
    landmark = landmarks[fn]

    for ldm in landmark:
        x = ldm[0] * w
        y = ldm[1] * h
        # Using cv2.circle() method 
        # Draw a circle with blue line borders of thickness of 2 px 
        face_img = cv2.circle(face_img, (int(x), int(y)), radius, color, thickness)
    face_imgs.append(face_img)
    
plot.display_multiple_images(face_imgs)

# %%
