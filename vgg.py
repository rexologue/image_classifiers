from torch.nn import Linear, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, Sequential, Module

from utils import Flatten

class FeatureVGG(Module):
    def __init__(self):
        super(FeatureVGG, self).__init__()
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(256, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Flatten()
        )
    
    def forward(self, x):
        return self.features(x)
        

class ClassifierVGG(Module):
    def __init__(self, num_classes):
        super(ClassifierVGG, self).__init__()

        self.model = FeatureVGG()

        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.model(x))
    