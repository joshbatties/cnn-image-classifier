import tensorflow as tf
import numpy as np
from model import ImageClassifier
import matplotlib.pyplot as plt

def load_and_preprocess_mnist():
    """Loads and preprocesses the MNIST dataset."""
    # Load MNIST dataset
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    X_train_full = X_train_full / 255.
    X_test = X_test / 255.
    
    # Split training set to create validation set
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
    
    # Add channel dimension
    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def plot_training_history(history):
    """Plots training history."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_preprocess_mnist()
    
    # Create and compile model
    model = ImageClassifier(input_shape=(28, 28, 1), num_classes=10)
    model.compile_model(learning_rate=0.001)
    
    # Train model
    history = model.train(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=10,
        batch_size=32
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save('mnist_classifier.h5')

if __name__ == "__main__":
    main()
