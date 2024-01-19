# %%
import torch

# %%
state_dict_cpu = torch.load("../../checkpoints/best_resnet_101_student_e56.pth", map_location="cpu")
state_dict_cpu

state_dict_gpu = torch.load("../../checkpoints/best_resnet_101_student_e56.pth")
state_dict_gpu

# %%
for name, param in state_dict_cpu.items():
    param = state_dict_gpu[name].data.cpu()
    pretrained_param = param.data
    print(name, (param == pretrained_param).all())

# %%
torch.save(state_dict_cpu, "../../checkpoints/best_resnet_101_student_e56_cpu.pth")

# %%
state_dict_cpu = torch.load("../../checkpoints/best_resnet_101_student_e56_cpu.pth")
state_dict_cpu

# %%
