import numpy as np

def load_mnist_data():
    from tensorflow.keras.datasets import mnist

    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # Make sure images have shape (28, 28, 1)
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    # Scale images to the [0, 1] range
    train_x = train_x.astype("float32")/255.0
    test_x = test_x.astype("float32")/255.0
    print(f"# Data shape: {train_x.shape}.")
    return (train_x, train_y), (test_x, test_y)