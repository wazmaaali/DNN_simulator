import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

class CNNModel(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self,input_channels,input_size) -> None:
        print("init")
        self.name = "CNNModel"
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        print("here,",input_size)
        if input_size == 28:
            print("here 28")
            self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjusted for 4x4 spatial size
        else:
            print("here 32")
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        print("before")
        # self.showParams()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4) #update this later
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def prunnedModel(self):
        print("here")
        parameters_to_prune = (
        (self.conv1, 'weight'),
        (self.conv2, 'weight'),
        (self.fc1, 'weight'),
        (self.fc2, 'weight'),
        (self.fc3, 'weight'),
    )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2,
        )
        # print(isinstance(self.conv1.weight, torch.nn.Parameter))
        # self.removePrunning()
        print("after")
        # self.showParams()
        return self
    
    def removePrunning(self):
        parameters_to_prune = (
            (self.conv1, 'weight'),
            (self.conv2, 'weight'),
            (self.fc1, 'weight'),
            (self.fc2, 'weight'),
            (self.fc3, 'weight'),
        )
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
    
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
    
        
