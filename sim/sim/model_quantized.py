import warnings
warnings.filterwarnings("ignore") 

from torchvision.models.quantization import (resnet18, resnet50, mobilenet_v2,
                                             ResNet18_QuantizedWeights, 
                                             ResNet50_QuantizedWeights, 
                                             MobileNet_V2_QuantizedWeights)

MODEL = {
    "resnet18": resnet18(weights = ResNet18_QuantizedWeights, quantize=True),
    "resnet50": resnet50(weights = ResNet50_QuantizedWeights, quantize=True), 
    "mobilenet_v2": mobilenet_v2(weights = MobileNet_V2_QuantizedWeights, quantize=True), 
}
