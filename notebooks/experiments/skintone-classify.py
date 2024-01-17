# %%
import stone
import os
import os.path as osp
import random
import time

# %%
img_dir = "../../data/interim/aligned_faces"
img_fns = os.listdir(img_dir)
img_fns_sample = random.sample(img_fns, 16)

# %%
t1 = time.time()
for fn in img_fns_sample:
    fp = osp.join(img_dir, fn)
    result = stone.process(fp, image_type='color')
t2 = time.time()
print(t2-t1)
print(result)

# %%
