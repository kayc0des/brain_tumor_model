import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# Load data
data = np.load('../src/image_data.npz')
X_data = data['images']

target = pd.read_csv('../src/csv/BrainTumor.csv')
Y = np.array(target['Class'])

scaler = MinMaxScaler(feature_range=(0, 1))

# Reshape X_data for efficient scaling (each row represents a flattened image)
reshaped_X = X_data.reshape(-1, 240 * 240 * 3)

# Apply MinMaxScaler to normalize the image data
X_data_scaled = scaler.fit_transform(reshaped_X)

# Reshape back to the original format
X_data_scaled = X_data_scaled.reshape(X_data.shape)

# One-hot encode the labels
Y_label = to_categorical(Y, num_classes=2)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data_scaled, Y_label, test_size=0.2, random_state=42)

# Create directories for saving data
train_dir = '../data/train'
test_dir = '../data/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


def save_images_and_labels(images, labels, folder):
    '''
    Save images and labels
    
    Param:
        - images: images to be saved
        - labels for the images
        - folder to save images
    
    Output:
        - saves images and labels
    '''
    img_dir = os.path.join(folder, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    label_file = os.path.join(folder, 'labels.csv')
    label_data = []

    for i, (img, label) in enumerate(zip(images, labels)):
        img_path = os.path.join(img_dir, f'image_{i}.png')
        image = Image.fromarray((img * 255).astype('uint8'))  # Convert to uint8 and rescale
        image.save(img_path)
        
        # Flatten label
        if label.ndim > 1:
            label = np.argmax(label)
        
        label_data.append([img_path, label])

    df = pd.DataFrame(label_data, columns=['image_path', 'label'])
    df.to_csv(label_file, index=False)


if __name__ == '__main__':
    save_images_and_labels(X_train, Y_train, train_dir)
    save_images_and_labels(X_test, Y_test, test_dir)

    print("Data saved successfully!")
