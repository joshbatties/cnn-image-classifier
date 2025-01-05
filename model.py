import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D
)

class ImageClassifier:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Builds and returns the CNN model architecture."""
        model = tf.keras.Sequential([
            Conv2D(32, kernel_size=3, padding="same",
                  activation="relu", kernel_initializer="he_normal",
                  input_shape=self.input_shape),
            Conv2D(64, kernel_size=3, padding="same",
                  activation="relu", kernel_initializer="he_normal"),
            MaxPool2D(),
            BatchNormalization(),
            Flatten(),
            Dropout(0.25),
            Dense(128, activation="relu", kernel_initializer="he_normal"),
            Dropout(0.5),
            Dense(self.num_classes, activation="softmax")
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compiles the model with given learning rate."""
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
    def train(self, X_train, y_train, validation_data=None, epochs=10, batch_size=32):
        """Trains the model on the provided data."""
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data."""
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """Makes predictions on new data."""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Saves the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """Loads a saved model from disk."""
        return tf.keras.models.load_model(filepath)

class TransferLearningClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10, base_model='xception'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model
        self.model = self._build_model()
        
    def _build_model(self):
        """Builds transfer learning model using pretrained base."""
        if self.base_model_name.lower() == 'xception':
            base_model = tf.keras.applications.Xception(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Base model {self.base_model_name} not supported")
            
        # Freeze base model
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(self.num_classes, activation="softmax")
        ])
        
        return model
    
    def unfreeze_top_layers(self, num_layers=30):
        """Unfreezes the top layers of base model for fine-tuning."""
        base_model = self.model.layers[0]
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
            
    def compile_model(self, learning_rate=0.001):
        """Compiles the model."""
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
    def train(self, train_dataset, validation_data=None, epochs=10):
        """Trains the model using the provided tf.data.Dataset."""
        return self.model.fit(
            train_dataset,
            validation_data=validation_data,
            epochs=epochs
        )
    
    def save(self, filepath):
        """Saves the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """Loads a saved model from disk."""
        return tf.keras.models.load_model(filepath)
