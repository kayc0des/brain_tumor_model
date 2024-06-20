# Brain Tumor Classification

This repository contains code and data for a brain tumor classification project. The goal of this project is to classify brain scans as either having a tumor (positive) or not having a tumor (negative) using a Convolutional Neural Network (CNN).

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

![Architecture](img/prediction.png)

## Conclusion

This brain tumor classification project demonstrates the effectiveness of CNNs in medical image analysis. With high training accuracy and good validation performance, the model shows promise for real-world application in brain tumor detection. Further testing and validation on larger datasets could help in improving the model's robustness and reliability.

---

For more details, refer to the scripts and documentation within the repository.
