import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DataPreprocessor:
    '''
    This class handles data preprocessing tasks
    '''

    def __init__(self, image_dir, target_dir, target_file, data_dir):
        '''
        Init method for the DataPreprocessor class

        Args:
            image_dir (str): Folder containing brain scan images
            target_dir (str): Folder containing target files
            target_file (str): CSV file with target labels
            data_dir (str): Folder to save processed data

        Returns:
            None
        '''
        if (num_files := len(os.listdir(image_dir))) == 0:
            raise ValueError(
                'The image folder has no images, please add image data')
        if (num_files := len(os.listdir(target_dir))) == 0:
            raise ValueError(
                'The target folder has no files, please add target data')
        
        target_file_path = os.path.join(target_dir, target_file)
        if not os.path.isfile(target_file_path):
            raise ValueError(
                f'The target file {target_file} does not exist in the target folder')

        if not os.path.isdir(data_dir):
            raise ValueError(f'The data directory {data_dir} does not exist')

        self.image_dir = image_dir
        self.target_dir = target_dir
        self.target_file_path = target_file_path
        self.data_dir = data_dir

    def build_image_as_np(self):
        '''
        Builds npz from image data
        
        Args:
            None

        Returns:
            np.array: Array of image data
        '''
        num_files = len(os.listdir(self.image_dir))
        
        images = []
        for i in range(1, num_files + 1):
            filename = f'Image{i}.jpg'
            image_path = os.path.join(self.image_dir, filename)
            image = np.asarray(Image.open(image_path))
            
            images.append(image)
        
        self.image_array = np.asarray(images)
        
        return self.image_array

    def extract_labels(self):
        '''
        Extracts target labels

        Args:
            None

        Returns:
            pd.Series: Target labels
        '''
        df = pd.read_csv(self.target_file_path)
        labels = df['Class']
        
        self.labels = labels.to_numpy()
        return self.labels

    def image_to_npz(self):
        '''
        Saves image data and labels to an npz file

        Args:
            None

        Returns:
            None
        '''
        if not hasattr(self, 'image_array'):
            raise ValueError(
                'Image array not built. Please call build_image_as_np first.')
        if not hasattr(self, 'labels'):
            raise ValueError(
                'Labels not extracted. Please call extract_labels first.')

        np.savez_compressed(
            os.path.join(self.data_dir, 'data.npz'), X=self.image_array, Y=self.labels)

class DataProcessor:
    '''
    This class handles loading data, applying a scaler to X, 
    one-hot encoding Y, and performing a train-test split
    '''

    def __init__(self, npz_file):
        '''
        Init method for the DataProcessor class

        Args:
            npz_file (str): Path to the .npz file containing X and Y

        Returns:
            None
        '''
        self.npz_file = npz_file
        self._load_data()

    def _load_data(self):
        '''
        Loads data from the .npz file

        Args:
            None

        Returns:
            None
        '''
        data = np.load(self.npz_file)
        self.X = data['X']
        self.Y = data['Y']

    def apply_scaler(self):
        '''
        Applies StandardScaler to X

        Args:
            None

        Returns:
            np.array: Scaled X
        '''
        scaler = MinMaxScaler()
        self.X_scaled = scaler.fit_transform(
            self.X.reshape(
                len(self.X), -1)).reshape(self.X.shape)

        return self.X_scaled

    def apply_one_hot_encoding(self):
        '''
        Applies one-hot encoding to Y

        Args:
            None

        Returns:
            np.array: One-hot encoded Y
        '''
        self.Y_encoded = tf.keras.utils.to_categorical(self.Y)
        return self.Y_encoded

    def perform_train_test_split(self, test_size=0.2, random_state=42):
        '''
        Performs a train-test split

        Args:
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Seed used by the random number generator

        Returns:
            tuple: Train and test splits for X and Y
        '''
        if not hasattr(self, 'X_scaled'):
            raise ValueError(
                'Data not scaled. Please call apply_scaler first.')
        if not hasattr(self, 'Y_encoded'):
            raise ValueError(
                'Labels not encoded. Please call apply_one_hot_encoding first.')

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_scaled, self.Y_encoded, test_size=test_size, random_state=random_state
        )

        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def save_data(self, train_dir='../data/train', test_dir='../data/test'):
        '''
        Saves the train and test data into separate folders
        
        Args:
            train_dir (str): Directory to save training data
            test_dir (str): Directory to save testing data
        
        Returns:
            None
        '''
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        np.savez_compressed(os.path.join(train_dir, 'train_data.npz'), X=self.X_train, Y=self.Y_train)
        np.savez_compressed(os.path.join(test_dir, 'test_data.npz'), X=self.X_test, Y=self.Y_test)


# Debug
data_pipeline = DataPreprocessor('images/', 'csv/', 'BrainTumor.csv', 'processed_data/')
data_pipeline.build_image_as_np()
data_pipeline.extract_labels()
data_pipeline.image_to_npz()

data_processor = DataProcessor('processed_data/data.npz')
data_processor.apply_scaler()
data_processor.apply_one_hot_encoding()
data_processor.perform_train_test_split()
data_processor.save_data()