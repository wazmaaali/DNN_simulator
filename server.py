import flwr as fl
import CNNutil as utils
import sys
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
import time
from CNNModel import CNNModel
from ResNet import ResNet
import argparse
import Constants as Constant
import torch
import torch.nn as nn
import warnings
from flwr.common import NDArrays, Scalar
from typing import Optional, Tuple, Dict
import torch.nn.utils.prune as prune
from typing import Callable
from YOLOModel import YOLOModel

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(">>>>>>>>> after warning")
accuracies = []
_times = []
losses = []

test_loader = ""

    
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(32),
        }
        return config

    return fit_config

def get_eval_fn(model: nn.Module):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    if Constant.MODEL_NAME != 'yolo':
        print("Constant.DATATYPE: ",Constant.DATATYPE)
        train_loader, test_loader = utils.selectDatatype(Constant.DATATYPE)
    # else:
    #     pred_result,all_ground_truths = model.collect_all_predictions_and_ground_truths('mydata/images/val', 'mydata/labels/val')


    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ):
        if Constant.MODEL_NAME != 'yolo':
            # utils.set_model_params(model,parameters)  # Update model with the latest parameters
            loss, accuracy = utils.test(model, test_loader)
            # _times.append(server_round)
        # else:
        #     loss, accuracy = utils.test(model,pred_result,all_ground_truths)
            return loss, {"accuracy": accuracy}
        else:
            return None

    return evaluate

    
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, *args, **kwargs):
        self.model = model  # This might be used for evaluation
        super().__init__(*args, **kwargs)
        self.all_losses = []
        self.all_accuracies = []
        self.all_training_time = []

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, num_examples = super().aggregate_fit(rnd, results, failures)
        # Collect losses from each client
        round_ttime = []
        for _, eval_res in results:
            if eval_res.status.message == "Success":  # Check if the evaluation was successful
                round_ttime.append(eval_res.metrics['training_time'])
        
        self.all_training_time.append(round_ttime)

        return aggregated_weights, num_examples

    def aggregate_evaluate(self, rnd, results, failures):
        # Call the base class's aggregate_evaluate to perform standard aggregation
        loss, accuracy = super().aggregate_evaluate(rnd, results, failures)
        # Extract and store losses and accuracies from each client
        round_losses = []
        round_accuracies = []
       
        for _, eval_res in results:

            if eval_res.status.message == "Success":  # Check if the evaluation was successful
                round_losses.append(eval_res.loss)
                round_accuracies.append(eval_res.metrics['accuracy'])

        # Append the collected metrics from this round to the overall lists
        self.all_losses.append(round_losses)
        self.all_accuracies.append(round_accuracies)

        return loss, accuracy


def callServer(model,test_loader = "None"):
    test_loader = test_loader
    start_time = time.time()   
    strategy = CustomFedAvg(
    model=model,  # Assuming 'model' is defined elsewhere
    # min_available_clients=3,
    evaluate_fn=get_eval_fn(model),
    # on_fit_config_fn=get_on_fit_config_fn(),
    fraction_fit=1,
    fraction_evaluate=1,
)
    fl.server.start_server(
            #server_address="localhost:"+ str(sys.argv[1]),
            server_address = "127.0.0.1:5040",
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=1),

        )
    # print("Type of all_losses:", type(strategy.all_losses))

    print("Losses Distribution:", strategy.all_losses[0])
    print("Centralized Accuracies:", strategy.all_accuracies[0])
    print("Centralized training time:", strategy.all_training_time[0])

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in seconds
    print(f"SERVER: Flower server time it took is {elapsed_time:.2f} seconds")
    
    accuracies.append(strategy.all_accuracies[0])
    losses.append(strategy.all_losses[0])
    _times.append(strategy.all_training_time[0])

    return elapsed_time, accuracies, _times , losses


import os

if __name__ == "__main__":
  
    # model = ResNet().to(DEVICE)

    #pruneFlag can either be yes or no
    #python3 server.py --yes for yes else no
    #python3 server.py --model_type resnet --data_type mnist --yes


    parser = argparse.ArgumentParser(description='Example of using argparse with a yes/no flag.')
    parser.add_argument('--yes', action='store_true', help='A yes flag.')
    parser.add_argument('--p_type', type=str, choices=[None, 'global', 'local','quant'], default=None, help='Specify the Prunning type: global, local or None for default.')

    parser.add_argument('--model_type', type=str, choices=[None, 'cnn', 'resnet', 'yolo'], default=None, help='Specify the model type: cnn, resnet, yolo, or None for default.')
    parser.add_argument('--data_type', type=str, choices=[None, 'mnist', 'cifar', 'video'], default=None, help='Specify the model type: cnn, resnet, yolo, or None for default.')

    args = parser.parse_args()
    pruneFlag = args.yes
    Constant.MODEL_NAME = args.model_type
    Constant.DATATYPE = args.data_type
    
     
    start_time = time.time()   
    model_initialized = False
    while not model_initialized:
        try:
            model = utils.selectModel(args.model_type)
            if model is not None:  # Assuming selectModel returns None on failure
                model_initialized = True
                if args.p_type == 'global':
                    
                    model = model.prunnedModel().to(DEVICE)
                    # model = model.quantizedModel().to(DEVICE)
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
        train_loader, test_loader =utils.selectDatatype(args.data_type)

        callServer(model,test_loader)


    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in seconds
    print(f"SERVER: Flower server time it took is {elapsed_time:.2f} seconds")






 




