import os
import numpy as np
from PIL import Image
import pandas as pd


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
            raise ValueError(
                f'The data directory {data_dir} does not exist')

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
        
        return labels

    def image_to_npz(self):
        '''
        Saves a Numpy array to an npz file

        Args:
            None

        Returns:
            None
        '''
        if not hasattr(self, 'image_array'):
            raise ValueError('Image array not built. Please call build_image_as_np first.')

        np.savez_compressed(os.path.join(self.data_dir, 'image_data.npz'), images=self.image_array)


# Debug
data_pipeline = DataPreprocessor('images/', 'csv/', 'BrainTumor.csv', 'processed_data/')
