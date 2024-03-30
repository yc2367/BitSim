"""
Fetch the pre-trained pytorch model
"""

from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2

MODEL = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "mobilenet_v2": mobilenet_v2,
}