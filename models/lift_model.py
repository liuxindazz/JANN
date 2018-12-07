import torch.nn as nn
import torch
import torchvision.models as models


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        # #load alexnet
        # alexn = models.alexnet(pretrained=True)
        # alexn = torch.load()
        # # remove last fully-connected layer
        # new_classifier = nn.Sequential(*list(alexn.classifier.children())[:-1])
        # alexn.classifier = new_classifier
        # self.feature = alexn

        #load placesnet
        alexn = models.__dict__['alexnet'](num_classes=365)
        checkpoint = torch.load('/1116/models/alexnet_places365.pth.tar', map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        alexn.load_state_dict(state_dict)
        new_classifier = nn.Sequential(*list(alexn.classifier.children())[:-1])
        alexn.classifier = new_classifier
        self.feature = alexn

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc9', nn.Linear(4096, 19))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.structured_classifier = nn.Sequential()
        self.structured_classifier.add_module('s_fc8', nn.Linear(4096, 19))

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        feature = self.feature(input_data)
        class_output = self.class_classifier(feature)
        structured_output = self.structured_classifier(feature)

        return class_output, structured_output
