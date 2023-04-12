# Importing Libraries

import torch
from torchvision import datasets, models
from torchvision import transforms as T
from torch import nn
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
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Read CSV files containig Filenames, Labels, and Suplementary Data
df = pd.read_csv(r"C:\Users\nishq\Downloads\DeepLesion\broadlesion.csv")

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

# Load The Trained Object Detector
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 9) 
# Load the Pre-trained Object Detector
model.load_state_dict(torch.load(r"C:\Users\nishq\OneDrive\Desktop\Model_DeepLesion_Prepped\model4_loss0.2555956244468689.pt"))

# Use the Object Detector in Evaluation Mode
model.eval()

# Copy the Model to GPU
device = torch.device("cuda")
model = model.to(device)

def collate_fn(batch):
    return tuple(zip(*batch))

# Create Data Loaders from the Datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)


# Convert Image to Tensor
transformation_from_image_to_tensor = T.Compose([T.ToTensor()])



# linear stretching of a given pixel value x_c, with respect to the maximum and minimum pixel values in the image (x_max, x_min), and scales it to a new range defined by l.
def linearStretching(x_c, x_max, x_min, l):
    return (l - 1) * (x_c - x_min) / (x_max - x_min)

# performs histogram mapping of a given histogram h to a new histogram with a specified range l.
def mapping(h, l):
    cum_sum = 0
    t = np.zeros_like(h, dtype=np.int)
    for i in range(l):
        # Calculate Cumulative Distribution
        cum_sum += h[i]
        t[i] = np.ceil((l - 1) * cum_sum + 0.5)
    return t

# calculates a metric for hue preservation enhancement based on the given parameters.    
def f(lam, h_i, h_u, l):
    h_tilde = 1 / (1 + lam) * h_i + lam / (1 + lam) * h_u
    t = mapping(h_tilde, l)
    d = 0
    for i in range(l):
        for j in range(i + 1):
            if h_tilde[i] > 0 and h_tilde[j] > 0 and t[i] == t[j]:
                d = max(d, i - j)
    return d

# performs hue preservation enhancement on a given channel of the input image.
def huePreservation(g_i, i, x_hat_c, l):
    g_i_f = g_i.flatten()
    i_f = i.flatten()
    x_hat_c_f = x_hat_c.flatten()
    g_c = np.zeros(g_i_f.shape)
    g_c[g_i_f <= i_f] = (g_i_f / i_f * x_hat_c_f)[g_i_f <= i_f]
    g_c[g_i_f > i_f] = ((l - 1 - g_i_f) / (l - 1 - i_f) * (x_hat_c_f - i_f) + g_i_f)[g_i_f > i_f]
    
    return g_c.reshape(i.shape)

# performs image fusion using Laplacian and intensity-based weighting.
def fusion(i):
    lap = cv2.Laplacian(i.astype(np.uint8), cv2.CV_16S, ksize=3)
    c_d = np.array(cv2.convertScaleAbs(lap))
    c_d = c_d / np.max(np.max(c_d)) + 0.00001
    i_scaled = (i - np.min(np.min(i))) / (np.max(np.max(i)) - np.min(np.min(i)))
    b_d = np.apply_along_axis(lambda x: np.exp(- (x - 0.5) ** 2 / (2 * 0.2 ** 2)), 0, i_scaled.flatten()).reshape(i.shape)
    w_d = np.minimum(c_d, b_d)
    
    return w_d

# the main function that enhances the input image using the implemented enhancement techniques.
def enhance(x, p1=2.0, p2=8):
    # Normalize Image
    x = x/255
    # Corner case the values of p1 and p2 to lie int the domain distribution
    p1 = round(float(p1), 4)
    p2 = int(p2)
    if p2 <2:
        p2 = 2

    # Distribute R, G and B channel. But due to Grayscale Image, all are equated as the 1 channel Data
    x_r = x_g = x_b = x

    # Calculate Min and Max value
    x_max = np.max(np.max(np.max(x)))
    x_min = np.min(np.min(np.min(x)))
    
    # Limit of Pixel Values 0-255
    l = 256

    # Stretch the Channels
    x_hat_r = linearStretching(x_r, x_max, x_min, l)
    x_hat_g = linearStretching(x_g, x_max, x_min, l)
    x_hat_b = linearStretching(x_b, x_max, x_min, l)

    # Illumination is calculated by weighted sum
    i = (0.299 * x_hat_r + 0.587 * x_hat_g + 0.114 * x_hat_b).astype(np.uint8)
    
    # Create Image Histogram
    h_i = np.bincount(i.flatten())
    h_i = np.concatenate((h_i, np.zeros(l - h_i.shape[0]))) / (i.shape[0] * i.shape[1])
    h_u = np.ones_like(h_i) * 1 / l
    
    # Weighted Average of Image Histogram
    result = minimize_scalar(f, method = "brent", args = (h_i, h_u, l))
    h_tilde = 1 / (1 + result.x) * h_i + result.x / (1 + result.x) * h_u
    t = mapping(h_tilde, l)
    g_i = np.apply_along_axis(lambda x: t[x], 0, i.flatten()).reshape(i.shape)
    
    # Apply Hue Preservation for Global Features
    g_r = huePreservation(g_i, i, x_hat_r, l)
    g_g = huePreservation(g_i, i, x_hat_g, l)
    g_b = huePreservation(g_i, i, x_hat_b, l)

    # Apply CLAHE  for Local Features
    clahe = cv2.createCLAHE(clipLimit=p1, tileGridSize=(p2,p2))
    l_i = clahe.apply(i)
    l_r = huePreservation(l_i, i, x_hat_r, l)
    l_g = huePreservation(l_i, i, x_hat_g, l)
    l_b = huePreservation(l_i, i, x_hat_b, l)

    # Estimate weights for global and Local Enhancements
    w_g = fusion(g_i)
    w_l = fusion(l_i)
    w_hat_g = w_g / (w_g + w_l)
    w_hat_l = w_l / (w_g + w_l)
    # Perform Weighted Average to Get the Final image for 3 channels
    y_r = w_hat_g * g_r + w_hat_l * l_r
    y_g = w_hat_g * g_g + w_hat_l * l_g
    y_b = w_hat_g * g_b + w_hat_l * l_b

    # Take mean of the 3 Channels
    y = np.dstack((y_r, y_g, y_b)).astype(np.uint8)
    # y = Image.fromarray(y)
    # Convert The Image to Tensor
    img = transformation_from_image_to_tensor(y)
    img = ((img[0] +img[1] + img[2])/3).unsqueeze(-3)
    return img
    

    # Regressor module to Predict the CLAHE Parameters
class Regressor(nn.Module):
    # Constructor to initialize Layers
    def __init__(self):
        super().__init__()
        # Conv2D Layer to convert the Gray Scale Image to RGB to Input Pretrained ResNet on ImageNet
        self.graytorgb = nn.Conv2d(1, 3, kernel_size=3, padding=1, stride=1)
        
        # Batch Normalization Layer
        self.bn1 = nn.BatchNorm2d(3)
        # ReLU activation
        self.relu1 = nn.ReLU()
        # Load the Pretrained ResNet18 Architecture
        self.arch1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        # Extract 32 Features from 512 ResNet-18 outputs
        self.arch1.fc = nn.Linear(512, 32)
        # ReLU activation
        self.relu2 = nn.ReLU()
        
        # 2 Fully Connected Layers to output Parameter a
        self.fc1a = nn.Linear(32, 12)
        # ReLU activation
        self.relu1a = nn.ReLU()
        self.fc2a = nn.Linear(12, 1)
        # ReLU activation
        self.relu2a = nn.ReLU()

        # 2 Fully Connected Layers to output Parameter b
        self.fc1b = nn.Linear(32, 12)
        # ReLU activation
        self.relu1b = nn.ReLU()
        self.fc2b = nn.Linear(12, 1)
        # ReLU activation
        self.relu2b = nn.ReLU()

    # Forward Method implementing Multi-objective Learning
    def forward(self, input):
        # GrayScale to 3 Channel
        x = self.graytorgb(input)
        # Batch Normalization, ReLU and feed to ResNet-18 Architecture 
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.relu2(self.arch1(x))
        
        # Input x to Predict Parameter a
        a = self.relu1a(self.fc1a(x))
        a = self.relu2a(self.fc2a(a))

        # Input x to Predict Parameter b
        b = self.relu1b(self.fc1b(x))
        b = self.relu2b(self.fc2b(b))

        # Return a, b
        return a, b

# Instantiate the CNN-based Deep Learning Model
reg = Regressor()
# Copy the model to GPU
reg =  reg.to(device)
params = [p for p in reg.parameters() if p.requires_grad]
# Initiate Optimizer
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

# Defining Custom Loss Function
def my_loss(output, target, imgs):
    # unpacking Ouput
    p1, p2 = output
    # apply Transformation to convert Image to Tensor
    imgs = [np.array(T.ToPILImage()(i)) for i in imgs]
    # Enhance Image
    e = [enhance(imgs[x], p1.cpu()[x], p2.cpu()[x]) for x in range(len(imgs))]

    # convert The Images back To tensors
    imgs = torch.cat(list(t(image).unsqueeze(-4).to(device) for image in imgs))
    e = torch.cat(list(image.unsqueeze(-4).to(device) for image in e))

    with torch.no_grad():
        # Instantiate method for calcualting mean Average precision
        m1 = MeanAveragePrecision()
        m2 = MeanAveragePrecision()
        # Use Object Detector to make predictions on both Original and Enhanced Images
        y = model(imgs)
        x =  model(e)
        # Calculate mAP Score
        m1.update(x, target)
        m2.update(y, target)
        # Return the Difference between the mAP score.
        return {"map": m1.compute()["map"] - m2.compute()["map"]}

# function to train 1 epoch
def train_one_epoch(model, optimizer, loader, device, epoch):
    # copy the models to the GPU
    model.to(device)
    reg.to(device)
    # Use the model in training Mode
    reg.train()
    # dict to stoer Losses from all batches
    all_losses = []
    # For every iteration of batches
    for images, targets in tqdm(loader):
        # Prepare Images and Labels
        images = torch.cat(list(image.unsqueeze(-4).to(device) for image in images))
        targets = [{k: torch.tensor([v]).to(device) for k, v in t.items() if k!='filename'} for t in targets]
        
        # Predict Parameters from Model
        l = reg(images)

        # Estimate Loss
        loss_value = my_loss(l, targets, images)
        losses = loss_value["map"]
        loss_value = losses.item()
        
        # Append mAP Score
        all_losses.append(loss_value)
                
        # Update Optimizer
        optimizer.zero_grad()
        # Loss is backpropagated
        losses.requires_grad = True
        losses.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']}, loss: {np.mean(all_losses)}")
    return np.mean(all_losses)
    
# function to evaluate 
def evaluate(reg, loader, device):
    # Initiate Method to calculate mean Average Precision
    m = MeanAveragePrecision()

    with torch.no_grad():
        # Use model to evaluation mode
        reg.eval()
        # Start with loss as 0
        loss=0
        # For eveery batch in the Loader
        for images, targets in tqdm(loader):
                # prepare the Images and Labels
                targets = [{k: torch.unsqueeze(torch.tensor(v), 0).to(device) for k, v in target.items() if  k!="filename"} for target in targets]
                images = torch.cat([torch.unsqueeze(image, 0).to(device) for image in images])
                # Predict the Parameters
                l = reg(images)
                # unpack the parameters
                p1, p2 = l
                # Transform the Images
                imgs = [np.array(T.ToPILImage()(i)) for i in images]
                e = [enhance(imgs[x], p1.cpu()[x], p2.cpu()[x]) for x in range(len(imgs))]
                e = [x.cuda() for x in e]
                # predict from the Model
                y = model(e)
                # Calculate mAP
                m.update(y, targets)
                loss_dict = m.compute()
                loss += loss_dict["map"]
        # Return the mean mAP of the batches
        return loss/len(loader)
       
train_score = []
valid_score = []
# Clip the Gradients
torch.nn.utils.clip_grad_norm_(reg.parameters(), 50)
# For each epoch iteration
for epoch in range(num_epochs):
    # Train the model
    train_score.append(train_one_epoch(reg, optimizer, train_loader, device, epoch))
    # Validation Score
    valid_score.append(evaluate(reg, valid_loader, device))
    print("Valid MAP: ", valid_score[-1])
    # If the score is better than the best
    if best_loss < valid_score[-1]:
        print("-"*50)
        print("Model Improved to ->", valid_score[-1])
        # Save the Model
        torch.save(model.state_dict(), f"C:\\Users\\nishq\\OneDrive\\Desktop\\model_enhanced{epoch}_loss{valid_score[-1]}.pt")
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