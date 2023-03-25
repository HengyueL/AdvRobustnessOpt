import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from robustness.cifar_models import resnet50 as rn_cifar50
from torchvision import models as torchvision_models
# ==== In Project Import ====
import sys
from models.blocks import ImageNetNormalizer, _straightThroughClamp
from models.robust_union_models import PreActResNet18
from models.deepmind_wideresnet import CIFAR100_MEAN, CIFAR100_STD, Swish, WideResNet


# ==== This Alexnetis used in the perceptual distance ====
class AlexNetFeatureModel(nn.Module):
    # def __init__(self, alexnet_model: torchvision_models.AlexNet):
    """
        Import the AlexNet pretrained ImageNet model.

        Note:
            This model can only calculate LPIPS and should not be used to train a classifier.
    """
    def __init__(self, lpips_feature_layer=False, use_clamp_input=False, device=None, dtype=torch.float):
        super().__init__()
        print(" >> Initing a ** AlexNet LPIPS ** model.")
        self.normalizer = ImageNetNormalizer(use_clamp_value=use_clamp_input)
        self.model = torchvision_models.alexnet(
            pretrained=True
            ).to(device, dtype=dtype).eval()

        assert len(self.model.features) == 13
        self.layer1 = nn.Sequential(self.model.features[:2])
        self.layer2 = nn.Sequential(self.model.features[2:5])
        self.layer3 = nn.Sequential(self.model.features[5:8])
        self.layer4 = nn.Sequential(self.model.features[8:10])
        self.layer5 = nn.Sequential(self.model.features[10:12])
        self.layer6 = self.model.features[12]
        self.lpips_feature_layer = lpips_feature_layer

    def features(self, x):
        x = x.float()
        x = self.normalizer(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer5 = self.layer5(x_layer4)

        feature_list = [x_layer1, x_layer2, x_layer3, x_layer4, x_layer5]
        if self.lpips_feature_layer:
            return_list = [feature_list[i] for i in self.lpips_feature_layer]
            return tuple(return_list)
        else:
            return tuple(feature_list)
    
    def classifier_features(self, x):
        x = self.normalizer(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer5 = self.layer5(x_layer4)
        return x_layer5

    def classifier(self, last_layer):
        x = self.layer6(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x
    
    def forward(self, x):
        """
        Returns logits for the given inputs.
        """
        return self.classifier(self.classifier_features(x))


class ImageNetResNetFeatureModel(nn.Module):
    def __init__(
        self, num_classes, use_clamp_input,
        pretrained=True, lpips_feature_layer=False
        ):
        """
            lpips_feature_layer is used to specify which latent feature is used to calculate the LPIPS distance.
            type ==> tuple
        """
        super().__init__()
        self.normalizer = ImageNetNormalizer(use_clamp_value=use_clamp_input)
        self.model = resnet50(pretrained=pretrained)
        self.num_classes = num_classes
        self.model.fc = nn.Linear(2048, self.num_classes)
        self.lpips_feature_layer = lpips_feature_layer

    def features(self, x):
        x = self.normalizer(x)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x_layer1 = x
        x = self.model.layer2(x)
        x_layer2 = x
        x = self.model.layer3(x)
        x_layer3 = x
        x = self.model.layer4(x)
        x_layer4 = x

        feature_list = [x_layer1, x_layer2, x_layer3, x_layer4]
        if self.lpips_feature_layer:
            return_list = [feature_list[i] for i in self.lpips_feature_layer]
            return tuple(return_list)
        else:
            return tuple(feature_list)

    def classifier_features(self, x):
        """
            This function is called to produce perceptual features.
            Output ==> has to be a tuple of conv features.
        """
        x = self.normalizer(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x_layer4 = x
        return x_layer4

    def classifier(self, last_layer):
        x = self.model.avgpool(last_layer)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        return x
    
    def forward(self, x):
        """
        Returns logits for the given inputs.
        """
        return self.classifier(self.classifier_features(x))


class CifarResNetFeatureModel(nn.Module):
    def __init__(self, num_classes, use_clamp_input,
        pretrained=False, lpips_feature_layer=False
    ):
        super().__init__()
        self.normalizer = ImageNetNormalizer(use_clamp_value=use_clamp_input)
        self.model = rn_cifar50()

    def classifier_features(self, x):
        x = self.normalizer(x)

        x = F.relu(self.model.bn1(self.model.conv1(x)))

        x = self.model.layer1(x)
        x_layer1 = x
        x = self.model.layer2(x)
        x_layer2 = x
        x = self.model.layer3(x)
        x_layer3 = x
        x = self.model.layer4(x, fake_relu=False)
        x_layer4 = x
        return (x_layer1, x_layer2, x_layer3, x_layer4)

    def classifier(self, last_layer):
        x = F.avg_pool2d(last_layer, 4)
        x = x.view(x.size(0), -1)
        x = self.model.linear(x)
        return x
    
    def forward(self, x):
        """
        Returns logits for the given inputs.
        """
        return self.classifier(self.classifier_features(x)[-1])


# ==== RobustUnion Model (Cifar10) ====
class UnionRes18(nn.Module):
    def __init__(
        self, use_clamp_input
    ):
        super().__init__()
        self.model = PreActResNet18()
        self.use_clamp_input = use_clamp_input
        self.clamp_layer = _straightThroughClamp.apply
    
    def forward(self, x):
        if self.use_clamp_input:
            x = self.clamp_layer(x)
        x = self.model(x)
        return x


# ==== DeepMind Adv Robustness Model ====
class DeepMindWideResNetModel(nn.Module):
    """
        Adv Trained Cifar10 WRN-70-16 model from:
        https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness
    """
    def __init__(self, use_clamp_input=False) -> None:
        super().__init__()
        self.use_clamp_input = use_clamp_input
        self.model = WideResNet(
            num_classes=10, depth=70, width=16,
            activation_fn=Swish, mean=CIFAR100_MEAN,
            std=CIFAR100_STD
        )
        self.clamp_layer = _straightThroughClamp.apply
    
    def forward(self, x):
        if self.use_clamp_input:
            x = self.clamp_layer(x)
        return self.model(x)

