# Image Data Processing

This folder contains scripts for processing image data, including converting images to a Numpy array format, normalizing the data, and saving it into training and testing sets. The processed data can be used for training machine learning models.

## Scripts

### 1. `preprocessing.py`

This script reads images from a specified folder and creates a Numpy array containing these images. It also saves the image data into a compressed `.npz` file.

#### Functionality:
- Extracts target data from a CSV file.
- Reads images from a folder and stores them in a Numpy array.
- Saves the image data to an `.npz` file.

#### Usage:
1. Ensure your images are named `Image1.jpg`, `Image2.jpg`, etc., and located in the `images/` folder.
2. Run the script to generate the `image_data.npz` file.

```bash
python3 preprocessing.py
```

### 2. `split.py`

This script loads the image data from the .npz file, scales it, one-hot encodes the labels, splits the data into training and testing sets, and saves them into specified directories.

#### Functionality:

-Loads and scales image data.
- One-hot encodes the labels.
- Splits the data into training and testing sets.
- Saves images and labels into ../data/train and ../data/test directories.

#### Usage:

- Ensure the image_data.npz file is located in the ../src/ directory.
- Ensure the labels are in the ../src/csv/BrainTumor.csv file.
- Run the script to process and save the data.

```bash
python3 split.py
```

## Directory Structure

- src/
    - csv/
        - BrainTumor.csv (CSV file containing labels)
    - images/
        - Image1.jpg, Image2.jpg, ... (Image files)
    - preprocessing.py (Script for creating the .npz file)
    - split.py (Script for processing and saving data)
    - image_data.npz

## Dependencies
- numpy
- pandas
- Pillow
- scikit-learn
- tensorflow

Install the required Python packages using pip:

```bash
pip install numpy pandas pillow scikit-learn tensorflow
```