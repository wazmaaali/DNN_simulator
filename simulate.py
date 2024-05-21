
import matplotlib.pyplot as plt
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
import Constants as Constant
import VideoFrameDataset as VideoFrameDataset
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from YoloDataset import YoloDataset
from ResNet import ResNet

import time
import torchvision
import torchvision.transforms as transforms
import warnings
from collections import OrderedDict
from torch.utils.data import random_split
import argparse
import CNNutil as utils
from server import callServer
# from client1 import callClient


def train_and_evaluate(model,train_loader=None,test_loader=None):
    elapsed_time, accuracies, _times , losses = callServer(model,test_loader)
    

    return elapsed_time, accuracies, _times , losses

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description='Example of using argparse with a yes/no flag.')
    parser.add_argument('--yes', action='store_true', help='A yes flag.')
    parser.add_argument('--p_type', type=str, choices=[None, 'global', 'local'], default=None, help='Specify the Prunning type: global, local or None for default.')
    parser.add_argument('--model_type', type=str, choices=[None, 'cnn', 'resnet', 'yolo'], default=None, help='Specify the model type: cnn, resnet, yolo, or None for default.')
    parser.add_argument('--data_type', type=str, choices=[None, 'mnist', 'cifar', 'video'], default=None, help='Specify the model type: cnn, resnet, yolo, or None for default.')

    

    args = parser.parse_args()
    # pruneFlag = args.yes
    Constant.DATATYPE =args.data_type
    Constant.MODEL_NAME = args.model_type
    print("nnnnn:",Constant.MODEL_NAME)
    model = utils.selectModel(args.model_type)  # Or another model
    if Constant.MODEL_NAME != 'yolo':
        print("inside")
        train_loader, test_loader =utils.selectDatatype(args.data_type)
    
    try:
    # Attempt to train and evaluate the unpruned model
        if Constant.MODEL_NAME != 'yolo':
            overall_time_unprunned, accuracy_unpruned, time_unpruned, losses_unprunned = train_and_evaluate(model, train_loader, test_loader)
        else:
            overall_time_unprunned, accuracy_unpruned, time_unpruned, losses_unprunned = train_and_evaluate(model)

    except Exception as e:
        print(f"Error training unpruned model: {e}")
        overall_time_unprunned, accuracy_unpruned, time_unpruned, losses_unprunned = None, None, None, None

    if overall_time_unprunned is not None:
        try:
            # Train and evaluate the globally pruned model
            # model_globally_pruned = global_prune(model_unpruned)
            model_globally_pruned = model.prunnedModel()
            if Constant.MODEL_NAME != 'yolo':
                overall_time_Gprunned, accuracy_Gpruned, time_Gprunned, losses_Gprunned = train_and_evaluate(model_globally_pruned, train_loader, test_loader)
            else:
                overall_time_Gprunned, accuracy_Gpruned, time_Gprunned, losses_Gprunned = train_and_evaluate(model_globally_pruned)

        except Exception as e:
            print(f"Error training globally pruned model: {e}")
            overall_time_Gprunned, accuracy_Gpruned, time_Gprunned, losses_Gprunned = None, None, None, None
    else:
        print("Skipping globally pruned model training due to unpruned model training failure.")

    if overall_time_unprunned is not None:
        try:
            # Train and evaluate the locally pruned model
            model_locally_pruned = model.localPrune()  # Assuming local_prune is a function that returns a pruned model
            if Constant.MODEL_NAME != 'yolo':
                overall_time_Lprunned, accuracy_Lpruned, time_Lprunned, losses_Lprunned = train_and_evaluate(model_locally_pruned, train_loader, test_loader)
            else:
                overall_time_Lprunned, accuracy_Lpruned, time_Lprunned, losses_Lprunned = train_and_evaluate(model_locally_pruned)

        except Exception as e:
            print(f"Error training locally pruned model: {e}")
            overall_time_Lprunned, accuracy_Lpruned, time_Lprunned, losses_Lprunned = None, None, None, None
    else:
        print("Skipping locally pruned model training due to unpruned model training failure.")

    # Optionally, continue with further processing or handling based on obtained results
    if overall_time_unprunned is not None and overall_time_Gprunned is not None and overall_time_Lprunned is not None:
        print("All models trained successfully. Proceeding with further analysis.")
    
                
        times = [overall_time_unprunned,overall_time_Gprunned, overall_time_Lprunned]  # Assuming these are average times or representative values

        print("unprunned")
        print(overall_time_unprunned, accuracy_unpruned, time_unpruned, losses_unprunned)
        
        print("Global")
        print(overall_time_Gprunned, accuracy_Gpruned, time_Gprunned, losses_Gprunned)
        
        print("Local")
        print(overall_time_Lprunned, accuracy_Lpruned, time_Lprunned, losses_Lprunned )
        
        
       # Sorting and selecting the top three values
        _accuracy_unpruned = accuracy_unpruned[0]
        _accuracy_Gpruned = accuracy_Gpruned[1]
        _accuracy_Lpruned = accuracy_Lpruned[2]

        # Sorting and selecting the top three values
        _losses_unprunned = losses_unprunned[0]
        _losses_Gprunned =losses_Gprunned[1]
        _losses_Lprunned = losses_Lprunned[2]

        # Sorting and selecting the top three values
        _time_unpruned = time_unpruned[0]
        _time_Gprunned = time_Gprunned[1]
        _time_Lprunned = time_Lprunned[2]
        
        # Sorting and selecting the top three values
        _accuracy_unpruned = sorted(_accuracy_unpruned)[-3:]
        _accuracy_Gpruned = sorted(_accuracy_Gpruned)[-3:]
        _accuracy_Lpruned = sorted(_accuracy_Lpruned)[-3:]

        # Sorting and selecting the top three values
        _losses_unprunned = sorted(_losses_unprunned)[:3]
        _losses_Gprunned = sorted(_losses_Gprunned)[:3]
        _losses_Lprunned = sorted(_losses_Lprunned)[:3]

        # Sorting and selecting the top three values
        _time_unpruned = sorted(_time_unpruned)[-3:]
        _time_Gprunned = sorted( _time_Gprunned)[-3:]
        _time_Lprunned = sorted(_time_Lprunned)[-3:]


        
        # DataFrames for accuracy, losses, and time
        df_accuracy = pd.DataFrame({
            'Unpruned': np.array(_accuracy_unpruned),
            'Globally Pruned': np.array(_accuracy_Gpruned),
            'Locally Pruned': np.array(_accuracy_Lpruned)
        })
        df_losses = pd.DataFrame({
            'Unpruned': np.array(_losses_unprunned),
            'Globally Pruned': np.array(_losses_Gprunned),
            'Locally Pruned': np.array(_losses_Lprunned)
        })
        df_time = pd.DataFrame({
            'Unpruned': np.array(_time_unpruned),
            'Globally Pruned': np.array(_time_Gprunned),
            'Locally Pruned': np.array(_time_Lprunned)
        })


        # Creating a figure with four subplots
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Adjusted for an extra plot

        # Plotting accuracy comparison
        axs[0].boxplot([df_accuracy['Unpruned'], df_accuracy['Globally Pruned'], df_accuracy['Locally Pruned']], labels=['Unpruned', 'Globally Pruned', 'Locally Pruned'])
        axs[0].set_title('Accuracy Comparison')
        axs[0].set_ylabel('Accuracy')

        # Plotting loss comparison
        axs[1].boxplot([df_losses['Unpruned'], df_losses['Globally Pruned'], df_losses['Locally Pruned']], labels=['Unpruned', 'Globally Pruned', 'Locally Pruned'])
        axs[1].set_title('Loss Comparison')
        axs[1].set_ylabel('Loss')

        # Plotting training time comparison
        axs[2].boxplot([df_time['Unpruned'], df_time['Globally Pruned'], df_time['Locally Pruned']], labels=['Unpruned', 'Globally Pruned', 'Locally Pruned'])
        axs[2].set_title('Training Time Comparison')
        axs[2].set_ylabel('Time (s)')

        # Adding a bar plot for overall times comparison
        categories = ['Unpruned', 'Globally Pruned', 'Locally Pruned']
        axs[3].bar(categories, times, color=['gray', 'blue', 'green'])
        axs[3].set_title('Overall Times')
        axs[3].set_ylabel('Time (s)')

        # Set an overall title and improve layout
        fig.suptitle('CNN Model Performance on CIFAR10 Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    else:
        print("Pruned model training was not conducted or failed.")
        
    
    
   

    
