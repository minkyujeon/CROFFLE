from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
# from torchvision.models import resnet50, resnet18
import torch
from .resnet import *

def get_backbone(backbone, castrate=False):
    if backbone == 'resnet18':
        backbone = resnet18(pretrained=False)
    elif backbone == 'resnet50':
        backbone = ResNet()
    # print('backbone:',backbone)
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):    
    if model_cfg.name == 'simsiam':
        model = SimSiam(ResNet())
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






