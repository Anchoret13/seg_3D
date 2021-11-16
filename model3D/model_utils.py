import torch.nn as nn
import MinkowskiEngine as ME
from model3D.Mink.minkunet import MinkUNet34C, MinkUNet14

def get_minkunet(in_channels, out_channels):
    return MinkUNet14(in_channels, out_channels)