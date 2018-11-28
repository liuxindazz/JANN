import torch.nn as nn
from functions import ReverseLayerF
import torchvision.models as models

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        alexn = models.alexnet(pretrained=True)
        # remove last fully-connected layer
        new_classifier = nn.Sequential(*list(alexn.classifier.children())[:-1])
        alexn.classifier = new_classifier
        self.feature = alexn

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(4096, 1000))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(1000))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(1000, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 19))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(4096, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        feature = self.feature(input_data)
        #feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
