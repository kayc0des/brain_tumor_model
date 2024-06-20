# Brain Tumor Classification

This repository contains code and data for a brain tumor classification project. The goal of this project is to classify brain scans as either having a tumor (positive) or not having a tumor (negative) using a Convolutional Neural Network (CNN).

A Convolutional Neural Network (CNN) is a type of deep learning algorithm specifically designed for processing structured grid data such as images. CNNs are particularly effective for image recognition and classification tasks due to their ability to automatically learn spatial hierarchies of features through the use of convolutional layers.

## Project Structure

The repository is structured as follows:

- `data/`: Contains the dataset used for model training.
  - `data_prep.ipynb`: Contains code to convert all 3762 brain scan images into a numpy array
  - `image_data.npz`: Numpy arrays of all 3762 brain scan images.
  - `csv/BrainTumor.csv`: CSV file with target labels for each image (0 for no tumor, 1 for tumor) - Class column.

- `model/`: Contains the saved model after training.

- `model.ipynb`: Contains the model development code

## Dataset

The dataset consists of 3762 brain scan images. The target labels are provided in `BrainTumor.csv` and have two classes:
- `0`: Absence of a tumor
- `1`: Presence of a tumor

The images were preprocessed and saved as numpy arrays in an `.npz` file for efficient loading and training.

## Model

The model used for this project is a Convolutional Neural Network (CNN) implemented using TensorFlow. The architecture of the final model is as follows:

![Architecture](img/architecture.png)

1. **Convolutional Layer 1:**
   - 32 filters of size 3x3
   - Activation: ReLU
   - Max Pooling: 2x2

2. **Convolutional Layer 2:**
   - 64 filters of size 3x3
   - Activation: ReLU
   - Max Pooling: 2x2

3. **Flatten Layer:**

4. **Fully Connected Dense Layer 1:**
   - 128 units
   - Activation: ReLU

5. **Fully Connected Dense Layer 2:**
   - 64 units
   - Activation: ReLU

6. **Output Layer:**
   - 2 units
   - Activation: Softmax

### Why These Components?

- **Convolutional Layers (Kernels):** Convolutional layers use filters (kernels) to scan through the input image and detect patterns such as edges, textures, and shapes. Each filter specializes in recognizing a specific feature, which helps the network learn spatial hierarchies of features in the images.

- **Max Pooling:** Max pooling layers reduce the spatial dimensions of the feature maps, retaining only the most important features while reducing computational load and controlling overfitting. By taking the maximum value in each patch of the feature map, max pooling also introduces a form of translation invariance, which is beneficial for image recognition.

- **ReLU Activation:** The Rectified Linear Unit (ReLU) activation function introduces non-linearity into the model, allowing it to learn complex patterns. ReLU also helps in preventing the vanishing gradient problem, where gradients become too small for effective training, by allowing gradients to flow more effectively during backpropagation.

- **Softmax Activation:** The softmax function is used in the output layer for multi-class classification. It converts the logits (raw predictions) into probabilities, providing a probability distribution over the classes, which makes it easier to interpret the model's predictions.

### Regularization and Training

- **L2 Regularization:** Applied to prevent overfitting by penalizing large weights.
- **Early Stopping:** Implemented to stop training when the validation performance stops improving.
- **Epochs:** The model completed training with 20 epochs.

## Training Performance

- **Accuracy:** 97.89%
- **Loss:** 0.1409

These metrics indicate that the model learned the training data well, achieving high accuracy and low loss.

## Validation Performance

- **Validation Accuracy:** 88.70%
- **Validation Loss:** 0.4164

![Output](img/output.png)

The validation accuracy is slightly lower than the training accuracy, which is typical and indicates good generalization. The higher validation loss compared to training loss is expected but still remains relatively low.

## Analysis

- **L2 Regularization:** Helped in penalizing large weights and encouraged the model to learn more generalizable features.

The results suggest that the CNN achieved good performance on the training data and generalizes well to unseen data. Further evaluation and testing might be needed to confirm the model's performance on completely new datasets.

## Prediction Example

![Prediction](img/Prediction.png)

## Conclusion

This brain tumor classification project demonstrates the effectiveness of CNNs in medical image analysis. With high training accuracy and good validation performance, the model shows promise for real-world application in brain tumor detection. Further testing and validation on larger datasets could help in improving the model's robustness and reliability.

---

For more details, refer to the scripts and documentation within the repository.
