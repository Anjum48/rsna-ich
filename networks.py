import numpy as np
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import timm


def model_builder(architecture_name, output=6):
    if architecture_name.startswith("resnet"):
        net = eval("models." + architecture_name)(pretrained=True)
        net.fc = torch.nn.Linear(net.fc.in_features, output)
        return net
    elif architecture_name.startswith("efficientnet"):
        n = int(architecture_name[-1])
        net = EfficientNet.from_pretrained(f'efficientnet-b{n}')
        net._fc = torch.nn.Linear(net._fc.in_features, output)
        return net
    elif architecture_name.startswith("densenet"):
        net = eval("models." + architecture_name)(pretrained=True)
        net.classifier = torch.nn.Linear(net.classifier.in_features, output)
        return net
    elif architecture_name == "vgg19":
        net = models.vgg19_bn(pretrained=True)
        net.classifier[6] = torch.nn.Linear(net.classifier[6].in_features, output)
        return net
    elif architecture_name == "seresnext":
        net = timm.create_model('gluon_seresnext101_32x4d', pretrained=True)
        net.fc = torch.nn.Linear(net.fc.in_features, 6)
        return net


# https://github.com/pudae/kaggle-hpa/blob/master/losses/loss_factory.py
def binary_focal_loss(gamma=2, **_):

    def func(input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss
        return loss.mean()

    return func


class Windowing(torch.nn.Module):
    def __init__(self, u=1, epsilon=1e-3, window_length=50, window_width=130, transform="sigmoid"):
        """
        Practical Window Setting Optimization for Medical Image Deep Learning https://arxiv.org/pdf/1812.00572.pdf
        :param u: Upper bound for image values, e.g. 255
        :param epsilon:
        :param window_length:
        :param window_width:
        """
        super(Windowing, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.u = u

        if transform == "sigmoid":
            weight = (2 / window_width) * np.log((u/epsilon) - 1)
            bias = (-2 * window_length / window_width) * np.log((u / epsilon) - 1)
            self.transform = self.sigmoid_transform
        else:  # Linear window
            weight = u / window_width
            bias = (-u / window_width) * (window_length - (window_width / 2))
            self.transform = self.linear_transform

        self.conv.weight = torch.nn.Parameter(weight * torch.ones_like(self.conv.weight))
        self.conv.bias = torch.nn.Parameter(bias * torch.ones_like(self.conv.bias))

    def linear_transform(self, x):
        return torch.relu(torch.max(x, torch.tensor(self.u)))

    def sigmoid_transform(self, x):
        return self.u * torch.sigmoid(x)

    def forward(self, img):
        return self.transform(self.conv(img))


class ResNetModel(torch.nn.Module):
    def __init__(self, step_train=False, output=6):
        super(ResNetModel, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, output)
        self.blocks = ["layer1", "layer2", "layer3", "layer4"]
        self.frozen_blocks = 4

        # Gradually unfreeze layers throughout training
        if step_train:
            for name, param in self.net.named_parameters():
                param.requires_grad_(False)
            self.unfreeze_layers()

    def phase1_model(self):
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, 1)

    def phase2_model(self):
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, 5)

    def unfreeze_layers(self, lower_bound=0):
        for name, param in self.net.named_parameters():
            if self.frozen_blocks < 0:
                param.requires_grad_(True)
            elif name.split(".")[0] in ["fc"]:
                param.requires_grad_(True)
            elif name.split(".")[0] in self.blocks[self.frozen_blocks:]:
                param.requires_grad_(True)

        if self.frozen_blocks >= lower_bound:
            self.frozen_blocks -= 1

    def forward(self, x):
        return self.net(x)


class DenseNetModel(torch.nn.Module):
    def __init__(self, step_train=False, output=6):
        super(DenseNetModel, self).__init__()
        self.net = models.densenet169(pretrained=True)
        self.net.classifier = torch.nn.Linear(self.net.classifier.in_features, output)
        self.blocks = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
        self.frozen_blocks = 4

        # Gradually unfreeze layers throughout training
        if step_train:
            for name, param in self.net.named_parameters():
                param.requires_grad_(False)
            self.unfreeze_layers()

    def phase1_model(self):
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, 1)

    def phase2_model(self):
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, 5)

    def unfreeze_layers(self, lower_bound=0):
        for name, param in self.net.named_parameters():
            if self.frozen_blocks < 0:
                param.requires_grad_(True)
            elif name.split(".")[0] in ["fc"]:
                param.requires_grad_(True)
            elif name.split(".")[0] in self.blocks[self.frozen_blocks:]:
                param.requires_grad_(True)

        if self.frozen_blocks >= lower_bound:
            self.frozen_blocks -= 1

    def forward(self, x):
        return self.net(x)


class EfficientNetModel(torch.nn.Module):
    """
    # Coefficients:   width,depth,res,dropout
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    """
    def __init__(self, n=0, step_train=False, output=6):
        super(EfficientNetModel, self).__init__()
        self.net = EfficientNet.from_pretrained(f'efficientnet-b{n}')
        self.net._fc = torch.nn.Linear(self.net._fc.in_features, output)

        filters = [block._block_args.output_filters for block in self.net._blocks]
        self.freeze_points = (np.where(np.diff(filters) > 0)[0])  # 6 main block groups which can be frozen/unfrozen
        self.frozen_blocks = 6

        # Gradually unfreeze layers throughout training
        if step_train:
            for name, param in self.net.named_parameters():
                param.requires_grad_(False)
            self.unfreeze_layers()

    def phase1_model(self):
        self.net._fc = torch.nn.Linear(self.net._fc.in_features, 1)

    def phase2_model(self):
        self.net._fc = torch.nn.Linear(self.net._fc.in_features, 5)

    def unfreeze_layers(self, lower_bound=3):
        try:
            fp = self.freeze_points[self.frozen_blocks]
        except IndexError:
            fp = np.Inf

        for name, param in self.net.named_parameters():
            if name.split(".")[0] in ["_conv_head", "_bn1", "_fc"]:
                param.requires_grad_(True)
            elif name.split(".")[1].isnumeric():
                block_number = int(name.split(".")[1])
                if block_number > fp:
                    param.requires_grad_(True)

        if self.frozen_blocks >= lower_bound:
            self.frozen_blocks -= 1
            print("Trainable blocks:", 6 - self.frozen_blocks)

    def forward(self, x):
        return self.net(x)
