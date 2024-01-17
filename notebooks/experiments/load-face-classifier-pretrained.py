# %%
import importlib

import torch
from torch import nn
import torch.nn.functional as F

import torchvision.models as models
from src.models.components import face_attrs_classifier
importlib.reload(face_attrs_classifier)

# %%
def load_pretrained(state_dict, pretrained_state_dict, debug=False):
    for name, pretrained_param in pretrained_state_dict.items():
        if name in state_dict.keys():
            param = state_dict[name].data
            pretrained_param = pretrained_param.data
            if param.shape == pretrained_param.shape:
                if debug:
                    print(name)
                param.copy_(pretrained_param)

    if debug:
        for name, param in pretrained_state_dict.items():
            if name in state_dict.keys():
                param = state_dict[name].data
                pretrained_param = pretrained_param.data

                if param.shape == pretrained_param.shape:                
                    if (param == pretrained_param).all():
                        print(name)
                
# %%
model = face_attrs_classifier.FaceAttrsClassifier(backbone="resnet101")

# %%
state_dict = model.state_dict()
for name, param in state_dict.items():
    print(name, param.data.shape)

# %%
pretrained_path = "../../checkpoints/best_resnet_101_student_e56.pth"
pretrained_state_dict = torch.load(pretrained_path)

# %%
for name, param in pretrained_state_dict.items():
    print(name, param.shape)
    
# %%
load_pretrained(state_dict, pretrained_state_dict, True)

# %%
