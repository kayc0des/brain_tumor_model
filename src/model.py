import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, accuracy_score

class Model:
    '''
    Model for Brain tumor classification.
    '''
    
    def __init__(self, input_shape=(240, 240, 3)):
        '''
        Initialization method.
        
        Args:
            input_shape: input shape for an image.
        '''
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        '''
        Build the CNN model.
        
        Args:
            input_shape: input shape for an image.
            
        Returns:
            model: the built CNN model.
        '''
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='relu', kernel_regularizer=L2(0.01)),
            Dense(units=64, activation='relu', kernel_regularizer=L2(0.01)),
            Dense(units=2, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, Y_train, validation_split=0.2, epochs=20, batch_size=32):
        '''
        Train the CNN model.
        
        Args:
            X_train: training data features.
            Y_train: training data labels.
            validation_split: proportion of training data to use for validation.
            epochs: number of epochs to train.
            batch_size: number of samples per gradient update.
            
        Returns:
            history: training history.
        '''
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            X_train, Y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        return history
    
    def save_model(self, filepath='../model/braintumor.pkl'):
        '''
        Save the trained model.
        
        Args:
            filepath: file path to save the model.
        '''
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='../model/braintumor.pkl'):
        '''
        Load a saved model.
        
        Args:
            filepath: file path to the saved model.
        '''
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No model found at {filepath}")
        
    def predict(self, X):
        '''
        Predict the classes for the given data.
        
        Args:
            X: input data.
            
        Returns:
            predictions: predicted classes.
        '''
        return self.model.predict(X)

    def evaluate(self, X_test, Y_test):
        '''
        Evaluate the model on the test data.
        
        Args:
            X_test: test data features.
            Y_test: test data labels.
        '''
        predictions = self.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(Y_test, axis=1)

        accuracy = accuracy_score(true_classes, predicted_classes)
        report = classification_report(true_classes, predicted_classes)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
