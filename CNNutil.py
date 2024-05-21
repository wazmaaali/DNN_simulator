from types import new_class
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from CNNModel import CNNModel
from YOLOModel import YOLOModel
import Constants as Constant
import VideoFrameDataset as VideoFrameDataset
from YOLOModel import YOLOLoss
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import json

import os

from ResNet import ResNet
from YoloDataset import YoloDataset

import time
import torchvision
import torchvision.transforms as transforms
import warnings
from collections import OrderedDict
from torch.utils.data import random_split
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.engine.results import Results
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.checks import check_imgsz
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics.utils.loss import VarifocalLoss




XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]


print(".....before warning")
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("after warning")
def selectModel(model_type=None):
    
    if model_type == 'resnet':
        print('ResNet model selected.')

        if Constant.DATATYPE == "mnist":
            model = ResNet(input_channels=1, num_classes=10).to(DEVICE)
        else:
            model = ResNet(input_channels=3, num_classes=10).to(DEVICE)
                    
        # Initialize and use your ResNet model here
    elif model_type == 'yolo':
        print('YOLO model selected.')
        model = YOLOModel().to(DEVICE)
        # Initialize and use your YOLO model here
    else:
        print('CNN model selected................')
        if Constant.DATATYPE == "mnist":
            print('CNN model mnist................')

            model = CNNModel(input_channels=1,input_size=28).to(DEVICE)
        else:
            model = CNNModel(input_channels=3,input_size=32).to(DEVICE)
            
        
        
    return model

def selectDatatype(data_type=None):
    
    if data_type == 'mnist':
        print('mnist selected.')   
        train_loader, test_loader = minist_load_data()   
    elif data_type == 'cifar':
        print('cifar selected.') 
        train_loader, test_loader = load_data()        
    # elif data_type == 'video':
    #     print('video selected.')   
    #     train_loader, test_loader = getVideo()#video_dataset()        
    else:
        train_loader, test_loader =load_data()   
        
        
    return train_loader, test_loader
def train(net, trainloader= None, epochs= None,pred_results= None,ground_truth=None):
    print("train")
    net.to(DEVICE)
    start_time=time.time()
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    if Constant.MODEL_NAME=="yolo":
        print("befo##########: ",pred_results)
        # all_predictions = get_allprediciton(pred_results)
        # pred_results = [sublist for sublist in pred_results if sublist]
        print("befo##########: ",pred_results)
        # all_boxes = []
        # if ground_truth == None:
        #     val_ground_truths = utils.get_ground_truths('mydata/images/train', 'mydata/labels/train')
        #     # Iterate over each key in the dictionary and extend the main list with these boxes
        #     for key in sorted(val_ground_truths.keys()):  # Sorting keys to maintain a consistent order, if needed
        #         all_boxes.append(val_ground_truths[key])  # Append each list of boxes to the main list

        # # Example of what `all_boxes` looks like now
        # print(all_boxes)
        # average_loss = calculate_loss(pred_results,all_boxes)
        # print(f"Average Training Loss: {average_loss:.4f}")
        # # accuracy = 0
        # torch.save(net.state_dict(), 'custom_yolo.pth')
 
    else: 
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(tqdm(trainloader), 1):
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels.to(DEVICE))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() 
            # Calculate the average training loss for the current epoch
        average_loss = running_loss / len(trainloader)        
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")
        print('Finished Training')
        elapsed_time = time.time() - start_time
        print(f'Training Time:  {elapsed_time:.3f} secs')
        # return average_loss  
   


def test(net, testloader = None,pred_results= None,ground_truth=None):
    print("test")
    net.to(DEVICE)
    """Validate the model on the test set."""
    correct, total, loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    # print("YOLO testloader: ",testloader)
    if Constant.MODEL_NAME=="yolo":
        print("befo##########: ",pred_results)
        all_predictions = get_allprediciton(pred_results)
        pred_results = [sublist for sublist in pred_results if sublist]
        print("befo#######afterrr###: ",pred_results)
        all_boxes = ground_truth
        if all_boxes == None:
            val_ground_truths = utils.get_ground_truths('mydata/images/val', 'mydata/labels/val')
            # Iterate over each key in the dictionary and extend the main list with these boxes
            for key in sorted(val_ground_truths.keys()):  # Sorting keys to maintain a consistent order, if needed
                all_boxes.append(val_ground_truths[key])  # Append each list of boxes to the main list

        # Example of what `all_boxes` looks like now
        print(all_boxes)
        average_loss = calculate_loss(pred_results,all_boxes)
        accuracy = 0      
    else:            
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels.flatten()).sum().item()
                total += labels.size(0)



        accuracy = correct / total
        average_loss = loss / len(testloader)
        print(f"Test Accuracy: {accuracy:.2f}")
    # print(f"Test Accuracy: {accuracy:.2f}, Average Test Loss: {loss / total:.4f}")

    
    print(f"Average Test Loss: {average_loss:.4f}")
    return average_loss, accuracy
   
def convert_to_yolo_labels(labels):
    # Mock function: in practice, you need to generate or have bounding box data
    return torch.full((len(labels), 5), fill_value=0.5)  # Mock [cx, cy, w, h, class]

def load_data() -> Dataset:
    """ Dataset download link:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3
    """
    
    print(">>>>>load CIFAR data")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)
    return trainloader, testloader

XY = tuple[np.ndarray, np.ndarray]
XYList = list[XY]

def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    # Shuffle data before partitioning to maintain correspondence
    X, y = shuffle(X, y)
    # Partition the shuffled data
    partitioned_X = np.array_split(X, num_partitions)
    partitioned_y = np.array_split(y, num_partitions)

    return list(zip(partitioned_X, partitioned_y))


def minist_load_data():
   # Load your training and test datasets here
    print( "dads : ",Constant.MODEL_NAME)
    if Constant.MODEL_NAME == 'cnn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    else:
        transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale images
])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
             
    train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(len(test_loader))
    return train_loader, test_loader


def calculate_loss(all_predictions, all_ground_truths):
    # Pseudocode: Actual implementation depends on your loss calculation
    all_ground_truths = convert_gtruthfloat(all_ground_truths)
    all_predictions = convert_gtruthfloat(all_predictions)
    # print(">>>>>>>all_ground_truths>>>>>>>>>: ",all_ground_truths)
    # print(">>>>>>>all_predictions>>>>>>>>>: ",all_predictions)

    total_loss = 0
    for pred, gt in zip(all_predictions, all_ground_truths):
        loss = getLoss(pred, gt)  # Define this function based on your model's specifics
        total_loss += loss
    average_loss = total_loss / len(all_predictions)
    print("Average Loss:", average_loss)

    return average_loss


def convert_gtruthfloat(all_ground_truths):
    converted_groundtruths = [
    [[int(item[0])] + list(map(float, item[1:])) for item in sublist]
    for sublist in all_ground_truths]
    return converted_groundtruths
    
    
def getLoss(all_predictions, all_ground_truths):
    
    
    pred = torch.tensor(all_predictions, dtype=torch.float32)
    gt = torch.tensor(all_ground_truths, dtype=torch.float32)

    # Extract bounding boxes and class labels
    pred_boxes = pred[:, 1:5] if pred.nelement() else torch.empty(0, 4)
    true_boxes = gt[:, 1:5] if gt.nelement() else torch.empty(0, 4)

    # Calculate loss only if there are boxes
    if pred_boxes.shape[0] > 0 and true_boxes.shape[0] > 0:
        # Pad the tensors to the same size
        max_boxes = max(pred_boxes.size(0), true_boxes.size(0))
        padded_pred_boxes = F.pad(pred_boxes, (0, 0, 0, max_boxes - pred_boxes.size(0)), "constant", 0)
        padded_true_boxes = F.pad(true_boxes, (0, 0, 0, max_boxes - true_boxes.size(0)), "constant", 0)

        # Calculate MSE loss
        box_loss = F.mse_loss(padded_pred_boxes, padded_true_boxes, reduction="mean")
    else:
        box_loss = torch.tensor(0.0)

    return box_loss.item()

def get_allprediciton(results):
    i=0
    j=0
    predictions = []
    for result in results:
        i = i+1
        for aa in result:
            j= j+1
            combined_data = [[int(aa.boxes.cls[k])] + aa.boxes.xywh[k].tolist() for k in range(aa.boxes.xywh.size(0))]
            predictions.append(combined_data)
    # print("get_allprediciton<<<<<<<<<<<<<<: ",get_allprediciton)
    return predictions

def get_ground_truths(image_dir, label_dir):
    """
    Fetches ground truths for a given directory of images and labels.
    
    Args:
    image_dir (str): Path to the directory containing images.
    label_dir (str): Path to the directory containing corresponding labels.
    
    Returns:
    dict: A dictionary where each key is an image file name and the value is the list of ground truths.
    """
    ground_truths = {}
    # List all files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    for image_file in image_files:
        # Corresponding label file path
        label_file = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as file:
                # Read all lines in the label file
                labels = file.readlines()
                # Parse each line to a list of floats
                labels = [list(map(float, line.strip().split())) for line in labels]
                ground_truths[image_file] = labels
        else:
            # No label file found for the image
            ground_truths[image_file] = []
    
    return ground_truths












