import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, number_of_filters, number_of_fc_features, number_of_classes):
        super(ConvNet, self).__init__()
        
        self.conv1 = self.conv_relu_maxpool(1, number_of_filters)
        self.conv2 = self.conv_relu_maxpool(number_of_filters, 2 * number_of_filters)
        self.conv3 = self.conv_relu_maxpool(2 * number_of_filters, 4 * number_of_filters)
        
        flattened_size = ((32 // 8) ** 2) * 4 * number_of_filters
        self.fc1 = self.fc_relu(flattened_size, number_of_fc_features)
        self.fc2 = self.fc(number_of_fc_features, number_of_classes)
    
    def conv_relu_maxpool(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def fc_relu(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
    
    def fc(self, in_features, out_features):
        return nn.Linear(in_features, out_features)
    
    def forward(self, x):
        max_pool1 = self.conv1(x)
        max_pool2 = self.conv2(max_pool1)
        max_pool3 = self.conv3(max_pool2)
        
        flattened = max_pool3.view(max_pool3.size(0), -1)
        
        fc1_output = self.fc1(flattened)
        logits = self.fc2(fc1_output)
        
        return logits

# Instantiate the model
number_of_filters = 16
number_of_fc_features = 256
number_of_classes = 10

model = ConvNet(number_of_filters, number_of_fc_features, number_of_classes)
print(model)
