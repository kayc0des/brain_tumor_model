import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X_scaled, self.Y_encoded, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, Y_train, Y_test


# Debug
data_processor = DataProcessor('processed_data/data.npz')
data_processor.apply_scaler()
data_processor.apply_one_hot_encoding()
X_train, X_test, Y_train, Y_test = data_processor.perform_train_test_split()
print(len(X_train))
print(len(X_test))
print(Y_train[:10])