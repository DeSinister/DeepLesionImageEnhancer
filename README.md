# Seeing the Unseen: Equipping Deep Learning to Enhance Images for Lesion Detection in CT Scans

This is a Course based project for CISC 881: Topics in Biomedical Imaging.
- The proposed research aims to propose an all-in-one solution for detecting lesions of varying size, severity and occurring locations from Full Body CT scan images with diversified noise and contrast levels. The final goal is to automate enhancing image quality and lesion detection via CNN-based Local and Global Enhancement.
- A ResNet-based Faster RCNN is trained on the CT scan images from DeepLesion Dataset. An additional ResNet18 CNN model is trained to predict parameters to enhance the Local and Global Contrast and remove underlying noise while preserving the image details.
- The image enhancement pipeline was able to enhance the performance from 30.77% mAP to 36.19% mAP. Thus, showing a significant improvement in lesion detection without retraining the pre-trained Object-Detector model.
- A Deep Learning based pipeline to enhance the local and global contrast of CT Scan images is proposed for the automatic detection of lesions with varying sizes and severity in different body locations. The experiment is validated on Deep Lesion dataset with a significant performance boost with the proposed pipeline. 

## Steps to Execute the Code
- Download the DeepLesion Dataset
- Set the path to the Dataset in both of files, and configure the Hyper-parameters
- First Run the Lesion_Detector_Final.py, this will produce an object detector model
- Copy the Model path and paste into Enhancer_Final.py file
- Run the Enhancer_Final.py file

References:
- https://github.com/pengyan510/glcae 
