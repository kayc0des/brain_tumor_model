import pandas as pd
import numpy as np
from PIL import Image
import os

# Extract target data
data = pd.read_csv('csv/BrainTumor.csv')
target = np.array(data['class'])

# Create Numpy Image Array
def create_image_array(folder_path):
    '''
    Read images from a folder and creates a Numpy array containing them
    Args:
        folder_path: Path to the folder containing images
    Returns:
        A Numpy array containing the loaded images
    '''
    
    # number of files in folder
    num_files = len(os.listdir(folder_path))
    
    images = []
    for i in range(1, num_files + 1):
        # Get full image path
        filename = f'Image{i}.jpg'
        image_path = os.path.join(folder_path, filename)
        image = np.asarray(Image.open(image_path))
        # Append the image to the list
        images.append(image)
    
    # Convert the image list to np.array
    image_array = np.asarray(images)

    return image_array


def save_to_npz(image_array, filename):
    '''
    Function saves a Numpy array to an npz file
    Args:
        image_array: The Numpy array to be saved
        filename: The filename for the npz file
    '''
    np.savez_compressed(filename, images=image_array)
    
    
if __name__ == '__main__':
    image_data = create_image_array('images/')
    save_to_npz(image_data, 'image_data.npz')