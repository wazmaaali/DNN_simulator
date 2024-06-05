import warnings
import flwr as fl
import numpy as np
import sys
import time
from CNNModel import CNNModel
from ResNet import ResNet
import argparse
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score
import torch
import torch.nn as nn
# import utils
import CNNutil as utils
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any  # Import the Any type
import Constants as Constant
from collections import OrderedDict


print("before warning")
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("after warning")
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description='Example of using argparse with a yes/no flag.')
    parser.add_argument('--yes', action='store_true', help='A yes flag.')
    parser.add_argument('--p_type', type=str, choices=[None, 'global', 'local','quant'], default=None, help='Specify the Prunning type: global, local or None for default.')

    parser.add_argument('--model_type', type=str, choices=[None, 'cnn', 'resnet', 'yolo'], default=None, help='Specify the model type: cnn, resnet, yolo, or None for default.')
    parser.add_argument('--data_type', type=str, choices=[None, 'mnist', 'cifar', 'video'], default=None, help='Specify the model type: cnn, resnet, yolo, or None for default.')

    args = parser.parse_args()
    pruneFlag = args.yes
    Constant.DATATYPE = args.data_type
    Constant.MODEL_NAME =args.model_type
    
    if(args.model_type != "yolo"):
        train_loader, test_loader = utils.selectDatatype(args.data_type)

     # Define Flower client
    class CifarClient(fl.client.NumPyClient):
      
        def __init__(self, model):
            self.model = model
            self.res=""
        def get_parameters(self, config):
            print("get_parameters")
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_model_params(self, parameters):
            print("set_parameters")

            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
        def fit(self, parameters, config):
            start_time = time.time()
            self.set_model_params(parameters)
            if Constant.MODEL_NAME == "yolo":                
                results,epochs = self.model.myTrain()
                ##for mAP averaged over IoU thresholds from 0.50 to 0.95.
                accuracy = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
                self.res = results
                updated_parameters = self.get_parameters(config)
                ##num_examples_processed=number of images in dataset×number of epochs                
                num_examples_processed = 46 * epochs
                print("Training accuracy of YOLO: ",accuracy)
            else:
                utils.train(self.model, train_loader, epochs=10)
                updated_parameters = self.get_parameters(config)
                num_examples_processed = len(train_loader.dataset) 
            
            training_duration = time.time() - start_time
            metrics = {"training_time":training_duration,"accuracy":accuracy}  # Add any additional metrics if needed
            return updated_parameters, num_examples_processed, metrics

        def evaluate(self, parameters, config):
            print("evaluate")
            
            if Constant.MODEL_NAME == "yolo":               
                results,epochs = self.model.valModel()
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<: ")
                ## GET loss, accuracy from results 
                # and fix following too 
                num_data = 12 * epochs
                pred_result,all_ground_truths = self.model.collect_all_predictions_and_ground_truths('mydata/images/val', 'mydata/labels/val')
                ##for mAP averaged over IoU thresholds from 0.50 to 0.95.
                accuracy = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
                loss, _ = utils.test(self.model,pred_result,all_ground_truths)
                print("Test accuracy: ", accuracy," loss: ",loss)
            else:
                loss, accuracy = utils.test(self.model, test_loader)
                num_data= len(test_loader.dataset)
                
            return loss, num_data, {"accuracy": accuracy}

    start_time = time.time()
    model_initialized = False

    while not model_initialized:
        try:
            model =  utils.selectModel(args.model_type)
            if model is not None:  # Assuming selectModel returns None on failure
                model_initialized = True
                if args.p_type == 'global':
                    model = model.prunnedModel()
                    print("inside prune flag")
                elif args.p_type == 'local':
                    model = model.localPrune()
                    print("local prunning")
                elif args.p_type == 'quant':
                    model = model.quantizedModel()
                    print("local prunning")        
                else:
                    print("no prunning")
                    pass
        
                print("Model successfully initialized.")
            else:
                print("Model initialization failed, retrying...")
        except Exception as e:  # Broad exception handling for illustration; specify your exception
            print(f"An error occurred during model initialization: {e}. Retrying...")
        # Optionally, introduce a delay here if needed

    if model_initialized:
        # Pass the initialized model to your CifarClient
        cifar_client = CifarClient(model=model)
        # Start the Flower client
        fl.client.start_numpy_client(
            server_address="127.0.0.1:5040",
            client=cifar_client,
        )

    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in seconds
    print(f"CLIENT1: Flower client1 time it took is {elapsed_time:.2f} seconds")





