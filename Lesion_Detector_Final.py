# Importing Libraries
import torch
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import os
from os import path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Read CSV files containig Filenames, Labels, and Suplementary Data
df = pd.read_csv(r"C:\Users\nishq\Downloads\DeepLesion\broadlesion.csv")

# Perform Exploratory Data Analysis
spacing = []
# append all the Spacing values 
for s in df["Spacing_mm_px_"]:
    spacing.append([float(x) for x in re.findall("-?\d+\.*\d*", s)])
# Print Spacing for Sample 1
print("Spacing for Sample 1: ", spacing[0])
# Transpose the values to Form 3 arays for respective x, y, and z direction
spacing = np.array(spacing)
spacing = np.transpose(spacing)

# For the 3 Diemensions, print the Minimum, Mean and Max Values
for i in spacing:
    print(f"{i} Min: {np.min(i)}, Mean: {np.mean(i)}, Max: {np.max(i)}") 


# Defining ToTensor object to convert Image to Tensor
t = T.ToTensor()

# Function To Extract Labels
def VOC(root, split):
    label_collection = []
    count = 0
    for i in range(len(df)):
        if df["Train_Val_Test"][i] in split and path.exists(root + df["File_name"][i]):
            label_collection.append({
                                    # "filename": df["File_name"][i],
                                    "filename": df["File_name"][i][:-8] + "\\" + df["File_name"][i][-7:], 
                                     "boxes": [float(x) for x in df["Bounding_boxes"][i].split(', ')],
                                     "labels": df["Coarse_lesion_type"][i]
                                     })
            count+=1
    return label_collection

# Class for the creating split from the Dataset
class LesionDetection(datasets.VisionDataset):
    # Constructor To initialise the target Labels Set 
    def __init__(self, root = f"C:\\Users\\nishq\\Downloads\\DeepLesion\\prep\\", split=[3], transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)
        self.split = split 
        self.root = root
        self.labels = VOC(root, self.split)

    # Load a particular Image with given ID
    def _load_image(self, id):
        id = self.labels[id]
        path = f"C:\\Users\\nishq\\Downloads\\DeepLesion\\minideeplesion\\" + id["filename"]
        # path =self.root + id["filename"]
        # Read the Image via CV2
        image = cv2.imread(path, 0)
        
        # Resize to 512, 512 Size
        if len(image) != 512:
            # Return the Scale to edit Label, and the Resized Image
            return len(image)/512, cv2.resize(image, (512, 512))
        
        # Print Image ID if Image not found for Debugging
        if type(image) == None:
            print(id)
        
        # If the Image size is 512 then return scale as 1
        return 1, image

    # Load a Target wioth given ID
    def _load_target(self, id):
        target = self.labels[id]
        return target
        
    # Load aprticular Set Image and Label
    def __getitem__(self, id):
        # Get particular Image and Label
        scale, image = self._load_image(id)
        target = self._load_target(id)
        # Edit Label based upon the Scale
        target['boxes'] = [int(x/scale) for x in target['boxes']]
        # Normalize and Convert eh Image to Tensor
        return t(Image.fromarray(image/255)), target
    
    # Returns the Length of the Data
    def __len__(self):
        return len(self.labels)

# Initiating the Training Dataset
train_dataset = LesionDetection(split=[3])
print(train_dataset)
# Initiating the Valdaition Dataset
valid_dataset = LesionDetection(split=[2])
print(valid_dataset)


# Hyper-parameters Configuration
batch_size = 26
num_epochs=100
lr=1.5e-5
weight_decay=1e-2
early_stop_patience = 20
best_loss = 0

# Instantiating the Object Detection Model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 9)


# Copy the Model to GPU
device = torch.device("cuda")
model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
# Initiate Optimizer
optimizer = torch.optim.Adam(params, lr=2e-5, weight_decay=1e-4)

def collate_fn(batch):
    return tuple(zip(*batch))

# Create Data Loaders from the Datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)


from torchmetrics.detection.mean_ap import MeanAveragePrecision
def nms(boxes, scores, iou_threshold=0.4, conf_threshold=0.5):
    """Performs Non-Maximum Suppression (NMS) on bounding boxes"""
    # Get indices of boxes sorted by descending order of scores
    indices = torch.argsort(scores, descending=True)
    # Initialize list to store selected boxes
    selected_boxes = []
    # Loop over boxes in descending order of scores
    while indices.numel() > 0:
        # Select box with highest score and add it to the list of selected boxes
        idx = indices[0]
        selected_boxes.append(idx)
        # Compute IoU between the selected box and the remaining boxes
        ious = calculate_iou(boxes[idx].unsqueeze(0), boxes[indices[1:]])
        # Remove indices of boxes with IoU greater than the threshold
        indices = indices[1:][ious < iou_threshold]
    # Return selected boxes
    selected_boxes = [selected_boxes[i] for i in range(len(selected_boxes)) if scores[i] > conf_threshold]
    print(selected_boxes)
    return selected_boxes

def calculate_iou(boxes1, boxes2):
    """Calculates Intersection over Union (IoU) between two bounding boxes"""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    # Calculate box areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # Calculate union area
    union = area1 + area2 - intersection
    # Calculate IoU
    iou = intersection / union
    return iou
def pick_one(boxes):
    return torch.argmax(boxes["scores"])

def train_one_epoch(model, optimizer, loader, device, epoch):
    # copy the models to the GPU
    model.to(device)
    # Use the model in training Mode
    model.train()
    # dict to store Losses from all batches
    all_losses = []
    all_losses_dict = []
    
    # For every iteration of batches
    for images, targets in tqdm(loader):
        # Prepare Images and Labels
        targets = [{k: torch.unsqueeze(torch.tensor(v), 0).to(device) for k, v in target.items() if  k!="filename"} for target in targets]
        images = torch.cat([torch.unsqueeze(image, 0).to(device) for image in images])
        
        # Predict from Model
        loss_dict = model(images, targets) 

        # Estimate Loss
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        # Append the Losses
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        
        # Update Optimizer
        optimizer.zero_grad()
        # Loss is backpropagated
        losses.backward()
        optimizer.step()
        
    # convert the 2D List to Pandas Dataframe
    all_losses_dict = pd.DataFrame(all_losses_dict)

    # print All the values
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))
    return np.mean(all_losses)


# function to evaluate 
def evaluate(model, loader, device):
    # Initiate Method to calculate mean Average Precision
    m = MeanAveragePrecision()
    with torch.no_grad():
        # Use model to evaluation mode
        model.eval()
        # Start with loss as 0
        loss=0
        # For eveery batch in the Loader
        for images, targets in tqdm(loader):
                # prepare the Images and Labels
                targets = [{k: torch.unsqueeze(torch.tensor(v), 0).to(device) for k, v in target.items() if  k!="filename"} for target in targets]
                images = torch.cat([torch.unsqueeze(image, 0).to(device) for image in images])
                # Predict from Model
                y = model(images)
                # Calculate mAP
                m.update(y, targets)
                loss_dict = m.compute()
                loss += loss_dict["map"]
        # Return the mean mAP of the batches
        return loss/len(loader)

# Clip the Gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

train_score = []
valid_score = []
# For each epoch iteration
for epoch in range(num_epochs):
    # Train the model
    train_score.append(train_one_epoch(model, optimizer, train_loader, device, epoch))
    # Validation Score
    valid_score.append(evaluate(model, valid_loader, device))
    print("Valid MAP: ", valid_score[-1])
    # If the score is better than the best
    if best_loss < valid_score[-1]:
        print("-"*50)
        print("Model Improved to ->", valid_score[-1])
        # Save the Model
        torch.save(model.state_dict(), f"C:\\Users\\nishq\\OneDrive\\Desktop\\model{epoch}_loss{valid_score[-1]}.pt")
        # Update the Best Valid Score and Reset Patience Early Stop
        best_loss = valid_score[-1]
        print("-"*50)
        early_stop_patience = 0
    else: 
        # If no improvement in Validation score
        early_stop_patience+=1
    # If No  improvement oiver given patience 
    if early_stop_patience == 20:
        break




# Plotting Training Loss with Green Color
plt.plot([x for x in range(1, len(train_score)+1)], train_score, 'g', label='Training loss')
# Setting Title
plt.title('Training Loss for applying Deep Learning based Image enhancement')
# Setting X and Y Labels
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Validation Loss with Blue Color  
plt.plot([x for x in range(1, len(valid_score)+1)], valid_score, 'b', label='Validation mAP Score')
# Setting Title
plt.title('Validation mAP Score for applying Deep Learning based Image enhancement')
# Setting X and Y Labels
plt.xlabel('Number of Epochs')
plt.ylabel('mAP Score')
plt.legend()
plt.show()