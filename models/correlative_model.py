import torch.nn as nn
import torch
import torchvision.models as models


class CorrelativeModel(nn.Module):

    def __init__(self):
        super(CorrelativeModel, self).__init__()
        #load alexnet
        alexn = models.alexnet(pretrained=True)
        # remove last fully-connected layer
        new_classifier = nn.Sequential(*list(alexn.classifier.children())[:-1])
        alexn.classifier = new_classifier
        print list(list(new_classifier.children())[1].parameters())
        self.feature = alexn

        # #load placesnet
        # alexn = models.__dict__['alexnet'](num_classes=365)
        # checkpoint = torch.load('models/alexnet_places365.pth.tar', map_location=lambda storage, loc: storage)
        # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        # alexn.load_state_dict(state_dict)
        # new_classifier = nn.Sequential(*list(alexn.classifier.children())[:-1])
        # alexn.classifier = new_classifier
        # self.feature = alexn


    def forward(self, input_data1, input_data2):
        input_data1 = input_data1.expand(input_data1.data.shape[0], 3, 224, 224)
        input_data2 = input_data2.expand(input_data2.data.shape[0], 3, 224, 224)
        RGB_output = self.feature(input_data1)
        Depth_output = self.feature(input_data2)

        return RGB_output, Depth_output
