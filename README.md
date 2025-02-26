# Early-pneumonia-detection
# PneumoNet: Chest X-ray Classification for Pneumonia Detection
# Overview
This project uses deep learning to build a classification model that detects pneumonia from chest X-ray images. The model is a Convolutional Neural Network (CNN) trained to classify X-ray images into two categories: Normal and Pneumonia. By automating the diagnosis process, the model helps speed up the detection of pneumonia, assisting healthcare professionals in early diagnosis and treatment.
 
# Purpose
Pneumonia is a severe lung infection that can be fatal if not detected early. Chest X-rays are commonly used in healthcare for diagnosing pneumonia, but manually analyzing large datasets of X-ray images can be time-consuming. This project aims to:

- Automate the detection of pneumonia: Use deep learning to classify chest X-ray images accurately.
- Assist healthcare professionals: Provide an efficient, accurate tool for diagnosing pneumonia based on X-ray images.
- Improve diagnostic speed: Help in quickly diagnosing patients with pneumonia, reducing the workload on radiologists and other healthcare professionals.

# Dataset
The dataset used in this project is the Chest X-ray Images dataset, which contains chest X-ray images labeled as Normal or Pneumonia. The dataset is divided into the following folders:

train: Contains training data (images of X-rays labeled as normal or pneumonia).
val: Contains validation data to evaluate model performance during training.
test: Contains test data to evaluate the final model after training.
The dataset consists of thousands of X-ray images, providing a robust foundation for training a deep learning model.


# How the Model Works
The model used in this project is a Convolutional Neural Network (CNN), which is widely used for image classification tasks. The CNN learns to extract important features from the images, such as edges, textures, and patterns, that distinguish pneumonia from normal chest X-rays.

Key Steps:
Preprocessing: The images are resized to 150x150 pixels, and pixel values are normalized to a range of [0, 1] to prepare them for input into the neural network.
Model Architecture: The model consists of several layers:
Convolutional Layers: These layers extract features from the images.
MaxPooling Layers: Reduce the spatial dimensions of the images to retain important features while reducing computational load.
Fully Connected Layers: Once features are extracted, fully connected layers classify the image into either the Normal or Pneumonia category.
Activation Functions:
ReLU (Rectified Linear Unit): Used in convolutional layers to introduce non-linearity.
Sigmoid: The final layer uses a sigmoid activation function to output a probability between 0 and 1, where 0 indicates Normal and 1 indicates Pneumonia.


# How the Model was Trained
Data Loading and Augmentation
The dataset was loaded using the image_dataset_from_directory function from TensorFlow. Data augmentation techniques (such as random flipping and rotations) can be applied during training to increase the robustness of the model.

Model Training
The model was trained using the Adam Optimizer with binary cross-entropy loss, which is suitable for binary classification tasks. The model was trained for 10 epochs, and the performance was evaluated using accuracy and loss metrics.

Evaluation
After training, the model's performance was evaluated on the test dataset to see how well it generalizes to unseen data.

# Visualizations
1. Accuracy Plot :
The Accuracy plot shows the training accuracy and validation accuracy over the course of training. Ideally, both accuracy values should increase and eventually stabilize, indicating that the model is learning well and is able to generalize to unseen data. A significant gap between the training and validation accuracy could indicate overfitting, where the model learns to perform very well on the training data but struggles with new, unseen data.

In our case, the training and validation accuracy have a similar trend, which suggests that the model is learning generalizable features. By the final epoch, the accuracy reaches around 85%, indicating decent performance.

Loss Plot
The Loss plot shows the training loss and validation loss over time. As training progresses, both losses should decrease. This indicates that the model is improving and getting better at classifying the images. A decreasing loss shows that the model is learning to minimize its error.

Our loss plot shows that both training and validation loss decreased steadily, with the final validation loss around 1.34. This suggests the model performed well, although there is potential for further improvement in the future.

2. Model Performance (Accuracy and Loss) :
After training, the model was evaluated on the test set, and the results were:
- Test Accuracy: The test accuracy represents the percentage of images that were correctly classified by the model from the test dataset. In this case, the test accuracy was 85.26%.
- Test Loss: The test loss is a measure of the model’s error on the test data. A lower value indicates better performance. Our model achieved a test loss of 1.34, which is relatively low.
These results indicate that the model is performing well on unseen data, correctly classifying around 85% of chest X-ray images.

3. Confusion Matrix and Classification Report :
For a more detailed understanding of the model's performance, especially on detecting Pneumonia, the Confusion Matrix and Classification Report provide valuable insights into how well the model performs on each class.

Confusion Matrix: This matrix shows the number of true positives, true negatives, false positives, and false negatives. This helps evaluate whether the model is confused between the two classes, Normal and Pneumonia.

A good model will have high values for true positives and true negatives, meaning it correctly identifies both pneumonia and normal images.

Classification Report: The classification report gives metrics such as Precision, Recall, and F1-Score for both classes. These metrics help in evaluating the model’s performance in a more detailed manner:

Precision: How many of the predicted pneumonia images are actually pneumonia?
Recall: How many actual pneumonia images are correctly identified by the model?
F1-Score: The harmonic mean of precision and recall. A high F1-score indicates that the model is balanced in terms of precision and recall.

# Model Evaluation Results
Test Accuracy: 85.26%
Test Loss: 1.34
These results indicate that the model is performing reasonably well in classifying chest X-ray images into Normal and Pneumonia categories. While there is room for improvement, these results are promising for a first iteration.

# Conclusion
The PneumoNet project demonstrates how deep learning can be applied to a critical problem in the medical field. By automating pneumonia detection from chest X-ray images, the model can assist healthcare professionals, speed up diagnosis, and ultimately save lives. The project also includes important visualization techniques to help evaluate and understand the model’s performance.






