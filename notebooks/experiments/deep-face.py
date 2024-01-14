# %%
import os
import os.path as osp
import random
import time

from deepface import DeepFace

# %%
img_dir = "../../../face-detect/yolov7/yolov7_face_test_20/cropped_face_imgs"
img_fns = os.listdir(img_dir)
img_fns_sample = random.sample(img_fns, 1)

# %%
t1 = time.time()
for fn in img_fns_sample:
    fp = osp.join(img_dir, fn)
    objs = DeepFace.analyze(img_path = fp, 
        actions = ['age', 'gender', 'race', 'emotion']
    )
t2 = time.time()
print(t2-t1)
