# Emotion Recognition Using Convolutional Neural Networks (CNN)

This project demonstrates an image-based emotion recognition model using a Convolutional Neural Network (CNN). The model is trained on the [Kaggle Emotion Recognition Dataset](https://www.kaggle.com) to classify emotions into six categories: Angry, Disgust, Fear, Happy, Sad, and Surprise.

## Dataset

The dataset consists of images categorized into six emotions. Images are resized to 64x64 pixels for training the model. The data is divided into training and validation sets, with 80% used for training and 20% for validation.

## Dependencies

This project requires the following libraries:

- `tensorflow`
- `keras`
- `opencv-python`
- `numpy`
- `joblib`

Install the required dependencies using:

```bash
pip install tensorflow keras opencv-python numpy joblib
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of:

1. **Conv2D layers**: Three convolutional layers with ReLU activation and MaxPooling2D to downsample the feature maps.
2. **Flatten layer**: Converts the 2D feature maps into a 1D vector.
3. **Dense layers**: Two dense layers, one with 128 units and ReLU activation, and a final output layer with softmax activation for classification.

The model is compiled using `categorical_crossentropy` as the loss function and `adam` optimizer.

### CNN Architecture Summary:

```
Layer (type)               Output Shape            Param #   
----------------------------------------------------------------
conv2d_1 (Conv2D)          (None, 62, 62, 32)      896       
max_pooling2d_1 (MaxPooling) (None, 31, 31, 32)    0         
conv2d_2 (Conv2D)          (None, 29, 29, 32)      9248      
max_pooling2d_2 (MaxPooling) (None, 14, 14, 32)    0         
conv2d_3 (Conv2D)          (None, 12, 12, 128)     36992     
max_pooling2d_3 (MaxPooling) (None, 6, 6, 128)     0         
flatten_1 (Flatten)        (None, 4608)            0         
dense_1 (Dense)            (None, 128)             589952    
dense_2 (Dense)            (None, 6)               774       
```

## Training the Model

The model is trained using the `flow_from_directory()` function from Keras' `ImageDataGenerator` for data augmentation. The training and validation data are generated from images in the dataset directory.

The model is trained for 25 epochs with the following parameters:
- Batch size: 32
- Epochs: 25
- Steps per epoch: Number of batches in the training set
- Validation steps: Number of batches in the validation set

## Performance

During training, the model achieves an accuracy of ~84% on the training set and ~73% on the validation set.

## Predictions

The model can predict the emotion in a new image by loading and preprocessing the image, then passing it through the model.

Example usage:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
test_image = image.load_img('path_to_image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make a prediction
result = cnn.predict(test_image)

# Get the predicted class
predicted_class = np.argmax(result[0])
print(f'Predicted Emotion: {predicted_class}')
```

## Saving the Model

The trained model is saved in two formats:
- **Joblib format**: `emotion_classification.pkl`
- **Keras H5 format**: `model.h5`

```python
import joblib
cnn.save('model.h5')
joblib.dump(cnn, 'emotion_classification.pkl')
```

## Running the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/emotion-recognition-cnn.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Make predictions using the trained model:
   ```bash
   python predict.py --image /path_to_image.jpg
   ```

## License

This project is licensed under the MIT License.
