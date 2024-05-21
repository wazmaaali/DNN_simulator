import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt



class ResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):  # Changed default to 1 for MNIST
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Input shape:", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("Post maxpool shape:", x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print("Pre-FC shape:", x.shape)
        return self.fc(x)
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    
    
    
    def prunnedModel(self):
        # Target the initial conv layer and the final fully connected layer for pruning
        parameters_to_prune = [
            (self.conv1, 'weight'),  # Prune weights of the first conv layer
            (self.fc, 'weight'),     # Prune weights of the final fully connected layer
        ]

        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2,
        )

        # Optionally, here you can also iterate through self.layer1 to self.layer4
        # to apply pruning to the conv layers within each BasicBlock
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                prune.l1_unstructured(block.conv1, name='weight', amount=0.2)
                prune.l1_unstructured(block.conv2, name='weight', amount=0.2)
                
        
        # self.removePrunning()
        return self
    def removePrunning(self):
        # Remove pruning from the initially specified layers
        prune.remove(self.conv1, 'weight')
        prune.remove(self.fc, 'weight')

        # Remove pruning from BasicBlock layers
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                prune.remove(block.conv1, 'weight')
                prune.remove(block.conv2, 'weight')
                
        print("after")
        # self.showParams()
        
    def localPrune(self):
        #iterative prunning
        for name, module in self.named_modules():
            # prune 20% of connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.2)
                # prune.remove(module, 'weight')
            # prune 40% of connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.4)
                # prune.remove(module, 'weight')

        return self
    
    def quantizedModel(self):
        model_int8 = torch.ao.quantization.quantize_dynamic(
        self,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)
        
        return self
    def showParams(self):   
        weights = []
        for param in self.parameters():
            weights += param.cpu().detach().numpy().flatten().tolist()

        
        plt.hist(weights, bins=100)
        plt.show()
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1)-> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out














# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.utils.prune as prune
# import matplotlib.pyplot as plt


# class ResNet(nn.Module):
#     def __init__(self, input_channels, num_classes=10):
#         super(ResNet, self).__init__()
#         self.name = "ResNetModel"
#         # Change from fixed 3 input channels to variable input_channels
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(64, 64, 3)
#         self.layer2 = self._make_layer(64, 128, 4, stride=2)
#         self.layer3 = self._make_layer(128, 256, 6, stride=2)
#         self.layer4 = self._make_layer(256, 512, 3, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
    
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(BasicBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layers.append(BasicBlock(out_channels, out_channels))
#         return nn.Sequential(*layers)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#         return self.fc(x)
    
    
#     def prunnedModel(self):
#         # Target the initial conv layer and the final fully connected layer for pruning
#         parameters_to_prune = [
#             (self.conv1, 'weight'),  # Prune weights of the first conv layer
#             (self.fc, 'weight'),     # Prune weights of the final fully connected layer
#         ]

#         # Apply global unstructured pruning
#         prune.global_unstructured(
#             parameters_to_prune,
#             pruning_method=prune.L1Unstructured,
#             amount=0.2,
#         )

#         # Optionally, here you can also iterate through self.layer1 to self.layer4
#         # to apply pruning to the conv layers within each BasicBlock
#         for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
#             for block in layer:
#                 prune.l1_unstructured(block.conv1, name='weight', amount=0.2)
#                 prune.l1_unstructured(block.conv2, name='weight', amount=0.2)
                
        
#         self.removePrunning()
#         return self
#     def removePrunning(self):
#         # Remove pruning from the initially specified layers
#         prune.remove(self.conv1, 'weight')
#         prune.remove(self.fc, 'weight')

#         # Remove pruning from BasicBlock layers
#         for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
#             for block in layer:
#                 prune.remove(block.conv1, 'weight')
#                 prune.remove(block.conv2, 'weight')
                
#         print("after")
#         # self.showParams()
        
#     def showParams(self):   
#         weights = []
#         for param in self.parameters():
#             weights += param.cpu().detach().numpy().flatten().tolist()

        
#         plt.hist(weights, bins=100)
#         plt.show()
    
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1)-> None:
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def forward(self,  x: torch.Tensor) -> torch.Tensor:
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out


















# class ResNet(nn.Module):
#     def __init__(self)-> None:
#         super(ResNet, self).__init__()
#         self.name = "ResNetModel"
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(64, 64, 3)
#         self.layer2 = self._make_layer(64, 128, 4, stride=2)
#         self.layer3 = self._make_layer(128, 256, 6, stride=2)
#         self.layer4 = self._make_layer(256, 512, 3, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, 10)
#         print("before")
#         # self.showParams()

#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(BasicBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layers.append(BasicBlock(out_channels, out_channels))
#         return nn.Sequential(*layers)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#         return self.fc(x)
    
    
#     def prunnedModel(self):
#         # Target the initial conv layer and the final fully connected layer for pruning
#         parameters_to_prune = [
#             (self.conv1, 'weight'),  # Prune weights of the first conv layer
#             (self.fc, 'weight'),     # Prune weights of the final fully connected layer
#         ]

#         # Apply global unstructured pruning
#         prune.global_unstructured(
#             parameters_to_prune,
#             pruning_method=prune.L1Unstructured,
#             amount=0.2,
#         )

#         # Optionally, here you can also iterate through self.layer1 to self.layer4
#         # to apply pruning to the conv layers within each BasicBlock
#         for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
#             for block in layer:
#                 prune.l1_unstructured(block.conv1, name='weight', amount=0.2)
#                 prune.l1_unstructured(block.conv2, name='weight', amount=0.2)
                
        
#         self.removePrunning()
#         return self
#     def removePrunning(self):
#         # Remove pruning from the initially specified layers
#         prune.remove(self.conv1, 'weight')
#         prune.remove(self.fc, 'weight')

#         # Remove pruning from BasicBlock layers
#         for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
#             for block in layer:
#                 prune.remove(block.conv1, 'weight')
#                 prune.remove(block.conv2, 'weight')
                
#         print("after")
#         # self.showParams()
        
#     def showParams(self):   
#         weights = []
#         for param in self.parameters():
#             weights += param.cpu().detach().numpy().flatten().tolist()

        
#         plt.hist(weights, bins=100)
#         plt.show()
    
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1)-> None:
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def forward(self,  x: torch.Tensor) -> torch.Tensor:
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out

