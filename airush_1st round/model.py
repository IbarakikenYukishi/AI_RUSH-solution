import torch
import torch.nn as nn
import torchvision.models as models


class Baseline(nn.Module):

    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet18(nn.Module):

    def __init__(self, out_size):
        super().__init__()
        model = models.resnet18(pretrained=True)
        model = list(model.children())[:-1]
        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

class Resnet50(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet50(pretrained=True)
        model = list(model.children())[:-1]
        model.append(nn.Conv2d(2048, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

class Alex(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, out_size)
        self.net = model

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

class Inception(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(2048, out_size)
        self.net = model

    def forward(self, image):
        return self.net(image)


class Vgg(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, out_size)
        self.net = model

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)
