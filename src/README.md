# Image Data Processing

This folder contains scripts for processing image data, including converting images to a Numpy array format, normalizing the data, and saving it into training and testing sets. The processed data can be used for training machine learning models.

## Folders

### 1. `csv/`

Contains the csv files with data relating to all 3762 brain scan images. The file used in this project is `BrainTumor.csv` because it contains a `Class` attribute or column.

### 2. `images/`

Contains all 3762 brain scan images

### 3. `processed_data/`

It holds the created `data.npz` file by the `preprocessing.py` script that contains the images  as numpy arrays and one hot encoder np.array for the target/labels.

## Scripts

### 1. `preprocessing.py`

This script is designed to handle data preprocessing and processing for brain tumor classification using a Convolutional Neural Network (CNN). The script includes two main classes: DataPreprocessor and DataProcessor, each responsible for different aspects of the data preparation pipeline.

### Classes and Functions

| Class             | Method                                    | Description                                                                                                             |
|-------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `DataPreprocessor`| `__init__(self, image_dir, target_dir, target_file, data_dir)` | Initializes the DataPreprocessor class.                                                                                 |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `image_dir` (str): Folder containing brain scan images.                                                               |
|                   |                                           | - `target_dir` (str): Folder containing target files.                                                                   |
|                   |                                           | - `target_file` (str): CSV file with target labels.                                                                     |
|                   |                                           | - `data_dir` (str): Folder to save processed data.                                                                      |
|                   | `build_image_as_np(self)`                 | Builds an array of image data from the specified directory.                                                             |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `np.array`: Array of image data.                                                                                      |
|                   | `extract_labels(self)`                    | Extracts target labels from the specified CSV file.                                                                     |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `pd.Series`: Target labels.                                                                                           |
|                   | `data_to_npz(self, filename)`             | Saves image data and labels to a compressed `.npz` file.                                                                |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `filename` (str): Name of the output file.                                                                            |
| `DataProcessor`   | `__init__(self, npz_file)`                | Initializes the DataProcessor class.                                                                                    |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `npz_file` (str): Path to the `.npz` file containing image data and labels.                                           |
|                   | `_load_data(self)`                        | Loads data from the specified `.npz` file.                                                                              |
|                   | `apply_scaler(self)`                      | Applies MinMaxScaler to the image data.                                                                                 |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `np.array`: Scaled image data.                                                                                        |
|                   | `apply_one_hot_encoding(self)`            | Applies one-hot encoding to the target labels.                                                                          |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `np.array`: One-hot encoded labels.                                                                                   |
|                   | `perform_train_test_split(self, test_size=0.2, random_state=42)` | Performs a train-test split on the data.                                                                                |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `test_size` (float): Proportion of the dataset to include in the test split.                                          |
|                   |                                           | - `random_state` (int): Seed used by the random number generator.                                                       |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `tuple`: Train and test splits for the image data and labels.                                                         |
|                   | `save_data(self, train_dir='../data/train', test_dir='../data/test')` | Saves the train and test data into separate directories.                                                                |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `train_dir` (str): Directory to save training data.                                                                   |
|                   |                                           | - `test_dir` (str): Directory to save testing data.                                                                     |


### 2. `model.py`

This script handles the construction, training, saving, loading, prediction, and evaluation of a Convolutional Neural Network (CNN) for brain tumor classification.

### Classes and Functions

| Class             | Method                                    | Description                                                                                                             |
|-------------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `Model`           | `__init__(self, input_shape=(240, 240, 3))` | Initializes the Model class.                                                                                           |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `input_shape` (tuple): Input shape for an image.                                                                     |
|                   | `build_model(self, input_shape)`          | Builds the CNN model.                                                                                                   |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `input_shape` (tuple): Input shape for an image.                                                                     |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `model` (Sequential): The built CNN model.                                                                           |
|                   | `train(self, X_train, Y_train, validation_split=0.2, epochs=20, batch_size=32)` | Trains the CNN model.                                                                                                  |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `X_train` (np.array): Training data features.                                                                         |
|                   |                                           | - `Y_train` (np.array): Training data labels.                                                                           |
|                   |                                           | - `validation_split` (float): Proportion of training data to use for validation.                                        |
|                   |                                           | - `epochs` (int): Number of epochs to train.                                                                            |
|                   |                                           | - `batch_size` (int): Number of samples per gradient update.                                                            |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `history` (History): Training history.                                                                                |
|                   | `save_model(self, filepath='../model/braintumor.pkl')` | Saves the trained model.                                                                                               |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `filepath` (str): File path to save the model.                                                                        |
|                   | `load_model(self, filepath='../model/braintumor.pkl')` | Loads a saved model.                                                                                                   |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `filepath` (str): File path to the saved model.                                                                       |
|                   | `predict(self, X)`                        | Predicts the classes for the given data.                                                                                |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `X` (np.array): Input data.                                                                                           |
|                   |                                           | **Returns:**                                                                                                            |
|                   |                                           | - `predictions` (np.array): Predicted classes.                                                                          |
|                   | `evaluate(self, X_test, Y_test)`          | Evaluates the model on the test data.                                                                                   |
|                   |                                           | **Args:**                                                                                                               |
|                   |                                           | - `X_test` (np.array): Test data features.                                                                              |
|                   |                                           | - `Y_test` (np.array): Test data labels.                                                                                |


### 3. `prediction.py`

This script handles the prediction of whether an image contains a tumor or not using a pre-trained model.

### Functions

| Function        | Description                                                                                              |
|-----------------|----------------------------------------------------------------------------------------------------------|
| `predict(image_path)` | Predicts whether an image contains a tumor or not using the pre-trained model.                        |
|                 | **Args:**                                                                                                  |
|                 | - `image_path` (str): Path to the image file.                                                              |
|                 | **Returns:**                                                                                               |
|                 | - `dict`: Dictionary containing prediction results and probabilities.                                      |
|                 | **Raises:**                                                                                                |
|                 | - `FileNotFoundError`: If the model file is not found.                                                     |
| `main`          | Command-line interface to predict tumor presence in an image.                                              |
|                 | **Args:**                                                                                                  |
|                 | - `image_path` (str): Path to the image file provided as a command-line argument.                           |
|                 | **Returns:**                                                                                               |
|                 | - Prints the prediction result to the console.                                                             |
