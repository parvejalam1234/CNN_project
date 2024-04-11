# CNN_project
Project Report: Image Classification of Cats and Dogs using Convolutional Neural Networks (CNN)

1. Introduction

This project aims to develop a Convolutional Neural Network (CNN) model to classify images of cats and dogs. Image classification is a fundamental task in computer vision, and CNNs have proven to be highly effective in this domain. The project utilizes TensorFlow and Keras libraries for model development and training.

2. Project Overview

Libraries Used: TensorFlow, Keras
TensorFlow Version: 2.12.0
Dataset: The dataset comprises two subsets:
Training Set: 8000 images
Test Set: 2000 images
Model Architecture: The CNN model architecture includes:
Convolutional layers with ReLU activation
Max pooling layers
Flattening layer
Fully connected layers with ReLU activation
Output layer with sigmoid activation
Training Parameters:
Optimizer: Adam
Loss function: Binary Crossentropy
Metrics: Accuracy
Epochs: 25
Data Preprocessing:
ImageDataGenerator for data augmentation and normalization
3. Data Preprocessing

Data preprocessing is essential for preparing the dataset before feeding it into the model. The following steps were performed:

Rescaling pixel values to the range [0,1]
Applying augmentation techniques such as shear range, zoom range, and horizontal flip to the training set
Rescaling pixel values for the test set
4. Model Building

The CNN model was constructed using TensorFlow and Keras. The model architecture consists of convolutional layers followed by max pooling layers, a flattening layer, fully connected layers, and an output layer. ReLU activation function was used for intermediate layers, and sigmoid activation function was used for the output layer to predict binary classes.

5. Model Training

The CNN model was compiled with the Adam optimizer, binary crossentropy loss function, and accuracy metrics. It was then trained on the training set and evaluated on the test set for 25 epochs. Training progress was monitored using validation data.

6. Results

The training process yielded the following results:

Training Accuracy: 86.64%
Validation Accuracy: 81.20%
Training Loss: 0.3093
Validation Loss: 0.4435
7. Making Predictions

A single prediction was made using an image of either a cat or a dog. The image was preprocessed and passed through the trained model to predict the class label.

8. Conclusion

The CNN model demonstrated promising performance in classifying images of cats and dogs. Further optimization and fine-tuning of the model architecture and training parameters could potentially improve its accuracy and generalization capabilities. Overall, the project highlights the effectiveness of CNNs in image classification tasks and provides a foundation for future enhancements and applications in computer vision.
