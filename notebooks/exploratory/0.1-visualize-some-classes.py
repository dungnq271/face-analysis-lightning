# %%
import os
import os.path as osp
from pathlib import Path

import pandas as pd
from src.reproducible_code.tools import plot

# %%
data_dir = "../../data/interim"
img_dir = osp.join(data_dir, "cropped_faces")
org_img_dir = osp.join(data_dir, "cropped_faces_groundtruth")
label_file = osp.join(data_dir, "labels.csv")

# %%
labels_df = pd.read_csv(label_file)
print(len(labels_df))
labels_df.head()

# %% [markdown]
## Races
race_counts = dict(labels_df.race.value_counts())
print(race_counts)
races = list(race_counts.keys())
races

# %%
for race in races:
    print(race)
    df = labels_df[labels_df["race"] == race].sample(16)
    fns = df.face_file_name.tolist()
    fps = [osp.join(img_dir, fn) for fn in fns]
    plot.display_multiple_images(fps)

# %% [markdown]
## Skintone
skintone_counts = dict(labels_df.skintone.value_counts())
print(skintone_counts)
skintones = list(skintone_counts.keys())
skintones

# %%
for skintone in skintones:
    print(skintone)
    df = labels_df[labels_df["skintone"] == skintone].sample(16)
    fns = df.face_file_name.tolist()
    fps = [osp.join(img_dir, fn) for fn in fns]
    plot.display_multiple_images(fps)

# %%
