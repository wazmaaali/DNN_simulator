import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from ultralytics import YOLO
import CNNutil as utils
import os

class YOLOModel(nn.Module):
    def __init__(self, model_source='yolov9c.pt'):
        super(YOLOModel, self).__init__()
        """
        Initialize a YOLO model either from a configuration file or pretrained weights.
        
        Parameters:
        - model_source (str): Path to a configuration file or pretrained weights.
        """
        print("YOlo init")
        self.model = self.load_model(model_source)
        
    def load_model(self, model_source):
        """
        Load the YOLO model from the specified source.
        
        This function is a placeholder and needs to be implemented according to
        how your specific YOLO model or framework expects models to be loaded.
        
        Parameters:
        - model_source (str): Path to a configuration file or pretrained weights.
        
        Returns:
        - model: Loaded model object.
        """

        print(f"Loading model from {model_source}")
        self.model = YOLO('yolov9c.pt')
        #sself.model.train()
        # self.model.info()
        return self.model
    
    
    def myTrain(self):
        epochs = 1
        results = self.model.train(data='dataset_config.yaml', epochs = epochs, imgsz=640,plots=True)
        return results, epochs
    
    def valModel(self):
        epochs = 1
        results = self.model.val(data='dataset_config.yaml', epochs=epochs, imgsz=640,plots=True)
        return results,epochs
    def predictModel(self,img_src):
        results = self.model.predict(source=img_src) # Display preds. Accepts all YOLO predict arguments
        return results
    
    def collect_all_predictions_and_ground_truths(self, img_dir, label_dir):
        all_predictions = []
        all_ground_truths = []
        image_files = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith('.jpg')]

        for img_file in image_files:
            # Generate predictions
            predictions = self.model.predict(source=img_file)
            all_predictions.append(predictions)

            # Assuming label files are in the same order and format
            label_file = img_file.replace('images', 'labels').replace('.jpg', '.txt')
            with open(label_file, 'r') as file:
                ground_truth = [line.strip().split() for line in file.readlines()]
                all_ground_truths.append(ground_truth)

        # print("all_predictions",all_predictions)
        # print("all_ground_truths",all_ground_truths)
        
        
        return all_predictions, all_ground_truths

   
    
    def prunnedModel(self):
        parameters_to_prune = []
        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.3,
        )
        
        # Optionally, make the pruning permanent
        # for module, _ in parameters_to_prune:
        #     prune.remove(module, 'weight')

        print("Global pruning applied")
        return self
    def localPrune(self):
        print("###################in local prune")

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv = module
                break

        # Apply L1 unstructured pruning to the first convolutional layer found
        if first_conv is not None:
            prune.l1_unstructured(first_conv, name='weight', amount=0.3)
            prune.remove(first_conv, 'weight')
            print("Pruning applied to:", name)
        else:
            print("No convolutional layer found for pruning")

        return self
    def precisionPlot():
        plot_metric(np.linspace(0, 1, len(self.model.box.p_curve.T)), self.model.box.p_curve.T, "Confidence", "Precision", "Precision-Confidence", save=True, save_dir="metrics_plots")

    def recallPlot():
        plot_metric(np.linspace(0, 1, len(self.model.box.r_curve.T)), self.model.box.r_curve.T, "Confidence", "Recall", "Recall-Confidence", save=True, save_dir="metrics_plots")

    def forward(self, x):
        output = self.model(x)
        return output

    def OldprunnedModel(self):
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                # We add tuples of (module, 'weight') for global pruning
                parameters_to_prune.append((module, 'weight'))
                if module.bias is not None:
                    parameters_to_prune.append((module, 'bias'))
        
        # Applying global unstructured pruning based on the L1 norm across all collected parameters
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.2,
        )
        self.removePrunning()
        return self
    
    def removePrunning(self):
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                prune.remove(module, 'weight')
                if module.bias is not None:
                    prune.remove(module, 'bias')
        
    
    
    
    def evalModel():
        img = Image.open(f'{work_dir}/runs/detect/train/P_curve.png')
        display(img)
        
    def getConfMatric():
        img = Image.open(f'{work_dir}/runs/detect/train/confusion_matrix_normalized.png')
        display(img)    
        
    def showResults():
        img = Image(filename='runs/detect/train/results.png', width=1000)  
        display(img)    
    def showParams(self):   
        weights = []
        for param in self.parameters():
            weights += param.cpu().detach().numpy().flatten().tolist()

        
        plt.hist(weights, bins=100)
        plt.show()
    
        

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5):
        """
        Initialize the YOLO loss module.

        Args:
            lambda_coord (float): Scaling factor for coordinate loss.
        """
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord

    def forward(self, pred_boxes, confidences, class_labels, targets):
        """
        Calculate the YOLO loss from predictions and targets.

        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes [batch_size, num_boxes, 4], where 4 = [x, y, w, h]
            confidences (torch.Tensor): Confidence scores for each bounding box [batch_size, num_boxes]
            class_labels (torch.Tensor): Predicted class probabilities [batch_size, num_boxes, num_classes]
            targets (torch.Tensor): Ground truth bounding boxes and class labels [batch_size, num_boxes, 4 + num_classes],
                                    assumes the first four values are bounding box coordinates and the rest are one-hot encoded class labels.

        Returns:
            torch.Tensor: Computed coordinate loss scaled by lambda_coord.
        """
        # Extract the ground truth coordinates
        true_coords = targets[..., :4]  # Assuming the first four are bounding box coordinates
        
        # Compute the MSE loss for coordinates
        coord_loss = F.mse_loss(pred_boxes, true_coords, reduction='sum')
        
        # Scale the coordinate loss
        coord_loss = self.lambda_coord * coord_loss

        # Optionally, you could add additional loss calculations here, such as:
        # - Objectness loss using confidences
        # - Classification loss using class_labels and targets[..., 4:]
        # These additional losses can be combined similarly to the coord_loss example above.

        return coord_loss