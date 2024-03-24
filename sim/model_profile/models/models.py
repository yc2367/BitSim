"""
Fetch the pre-trained pytorch model
"""

from torchvision.models import resnet18, resnet34, resnet50, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, vgg11_bn, vgg13_bn, vgg16_bn

MODEL = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "efficientnet_b0": efficientnet_b0,
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_S": mobilenet_v3_small,
    "mobilenet_v3_L": mobilenet_v3_large,
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn
}