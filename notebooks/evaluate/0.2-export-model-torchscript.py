# %%
import torch
import torch.nn as nn

from src.models.fashion_module import FashionLitModule

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
ckpt_path = "checkpoints/23_12_25_best.ckpt"
pth_path = "checkpoints/23_12_25_best.pth"

# %% [markdown]
## Load module and convert to TorchScript
model = FashionLitModule.load_from_checkpoint(ckpt_path)
model

# %%
script = model.to_torchscript()

print(f"Export {ckpt_path} to {pth_path}")
# save for use in production environment
torch.jit.save(script, pth_path)

# %% [markdown]
# ## Load pth model
model = torch.jit.load(pth_path).to(device)
model.eval()

# %%
inp = torch.rand(1, 3, 224, 224).to(device)
output = model(inp)
print(output)
