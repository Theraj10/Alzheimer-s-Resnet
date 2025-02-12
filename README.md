# Alzheimer-s-Resnet
This project utilizes TensorFlow and ResNet50 to classify Alzheimer's disease stages from MRI images. The dataset is preprocessed using ImageDataGenerator, and the model is fine-tuned for better performance


Dataset

Training Data: Augmented Alzheimer's Dataset

Validation Data: Original Alzheimer's Dataset

Images are resized to 224x224 for compatibility with ResNet50.

Dependencies

Ensure you have the following dependencies installed:

pip install tensorflow keras numpy matplotlib

Model Architecture

Uses ResNet50 (pre-trained on ImageNet) as the base model.

Freezes base model layers to retain pre-trained features.

Adds custom fully connected layers with ReLU activation.

Final output layer with softmax activation for 4-class classification.

Training Process

Data augmentation with ImageDataGenerator.

Uses categorical crossentropy as the loss function.

Optimized using Adam optimizer with a learning rate of 0.0001.

Trained for 4 epochs with batch size 32.

How to Run

Prepare the dataset: Place the training and validation images in the specified directories.

Run the script:

python train_model.py

Model Training: The model will train and save the best weights as resnet_alzheimer_model.h5.

Evaluate the model: The script will output test loss and test accuracy.

Model Saving & Loading

After training, the model is saved as:

model.save('resnet_alzheimer_model.h5')

To load the model later:

from tensorflow.keras.models import load_model
loaded_model = load_model('resnet_alzheimer_model.h5')

Results

The trained model is evaluated on a separate test dataset to determine its accuracy in classifying different stages of Alzheimer's disease.
