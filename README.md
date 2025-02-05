# DeepFake Detection using VGG16
## Overview
The DeepFake Detection project focuses on identifying and analyzing deepfake videos using advanced machine learning techniques. As deepfake technology evolves, it poses significant risks, such as spreading misinformation, manipulating public opinion, and infringing on personal privacy. The impact of deepfakes can be profound, leading to a loss of trust in media and potential harm to individuals whose likenesses are misused. By developing effective detection tools, this project aims to combat these challenges, helping to create a safer digital environment where people can trust the authenticity of the content they consume.
![image](https://github.com/user-attachments/assets/4c3ce79c-b253-455e-92ab-94a2b95a91f0)

## Technnologies Used
- Python
- Keras
- Tensorflow
- Scikit-learn
- OpenCv

## Dataset Used
We used the DFDC (DeepFake Detection Challenge) Dataset created by Meta, it includes thousand of labelled real and fake videos. We have divided this dataset into three parts - Test, Train and Val.

## Preprocessing Steps

1. **Frame Extraction**: Videos are processed to extract individual frames, allowing the model to analyze each frame as a separate image. This step helps in focusing on the visual content of the video.

2. **Resizing**: Extracted frames are resized to a consistent dimension (e.g., 224x224 pixels) to ensure uniformity in input size for the model. This is important for maintaining the aspect ratio and reducing computational load.

3. **Normalization**: Pixel values of the images are normalized (scaled to a range of 0 to 1) to improve the convergence of the model during training. This step helps the model learn more effectively by ensuring that all input features contribute equally.

4. **Data Augmentation**: To enhance the robustness of the model, various data augmentation techniques are applied, such as random rotations, flips, and brightness adjustments. This helps in increasing the diversity of the training dataset and reduces overfitting.

5. **Face Detection**: A face detection algorithm, specifically the **MTCNN (Multi-task Cascaded Convolutional Networks)**, is applied to identify and crop the faces in the frames. This focuses the model's attention on the most relevant part of the image, which is crucial for detecting deepfakes.

6. **Labeling**: Finally, the frames are labeled based on whether they contain real or deepfake content. This labeling is essential for supervised learning, allowing the model to learn from examples.

## Model Architecture (VGG16)

In our DeepFake Detection project, we utilize the **VGG16** architecture, a deep convolutional neural network known for its simplicity and effectiveness in image classification tasks. Developed by the Visual Geometry Group at the University of Oxford, VGG16 is characterized by its use of small convolutional filters (3x3) and a deep network structure, consisting of 16 layers with learnable weights.
#### Key Features of VGG16:
- **Convolutional Layers**: The network consists of multiple convolutional layers that extract features from the input images. Each convolutional layer is followed by a Rectified Linear Unit (ReLU) activation function, which introduces non-linearity into the model.
  
- **Pooling Layers**: After a series of convolutional layers, max pooling layers are used to down-sample the feature maps, reducing their spatial dimensions while retaining the most important information. This helps in reducing the computational load and controlling overfitting.

- **Fully Connected Layers**: The final layers of the network are fully connected layers that perform the classification task. The output layer uses a softmax activation function to produce probabilities for each class (real or deepfake).

- **Transfer Learning**: VGG16 is often pre-trained on a large dataset, ImageNet, allowing it to use learned features for new tasks. In our project, we fine-tune the pre-trained model on our specific dataset of real and deepfake videos to improve classification performance.

![VGG16 Architecture](https://github.com/user-attachments/assets/a3e202cf-d3d6-417a-bf76-22e8137029bb)


## Training Process

The dataset was split to ensure robust model training and evaluation. We employed an 80-10-10 split ratio, allocating 80% of the data for training, 10% for validation, and 10% for testing. To maintain balanced class distribution, we ensured an equal number of real and fake images in each set. The training data was enriched through various augmentation techniques, including random horizontal flips, rotations between -30° and 30°, and color enhancements ranging from 0.8x to 1.2x intensity. This augmentation process helps improve the model's robustness and generalization capabilities by introducing controlled variations in the training data.

## Result 
The final model achieved an accuracy of over 96% on the validation set. The confusion matrix from the model testing shows high precision and recall rates.
![model_accuracy](https://github.com/user-attachments/assets/37422e30-51c1-45d8-be2b-127357a57831)
![model_loss](https://github.com/user-attachments/assets/42594b29-a7f3-45ad-be04-dc2234c73acd)
![confusion_matrix](https://github.com/user-attachments/assets/4f58e96b-7c1d-4a30-bc85-90e9952f3261)
![Screenshot (3)](https://github.com/user-attachments/assets/806fbc71-8f51-4781-9873-61cb87e02246)

