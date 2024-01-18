# %%
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
print(root_dir)

# %%
data_dir = root_dir / "data/interim"

# %%
df = pd.read_csv(str(data_dir / "labels.csv"))
print(len(df))

# %%
img_dir = Path("aligned_faces/aligned_faces/images")
df["face_file_path"] = df["face_file_name"].apply(lambda x: str(img_dir / x))
df.head()

# %%
df_adt = pd.read_csv(str(data_dir / "labels_additional.csv"))
print(len(df_adt))

# %%
img_dir = Path("aligned_faces_additional/aligned_faces_additional/images")
df_adt["face_file_path"] = df_adt["face_file_name"].apply(lambda x: str(img_dir / x))
df_adt.head()

# %%
df_full = pd.concat([df, df_adt], axis=0).reset_index(drop=True)
print(len(df_full))
df_full.head()

# %%
df_full.to_csv(str(data_dir / "labels_full.csv"), index=None)

# %%
