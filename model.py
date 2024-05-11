import torch.nn as nn 
from torchvision import models
import torch


# mode output=4096
def MVGG():
    vgg = models.vgg11(pretrained = True)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-3])
    model = vgg    
    return model        
