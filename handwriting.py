import sys

import tensorflow as tf


def prepare_data():
    mnist = tf.keras.datasets.mnist
    # Prepare data for training
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # bring values between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
    )
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
    )
    print("loaded data")
    return mnist, x_train, y_train, x_test, y_test


def train_model(x_train, y_train):
    # TODO(students): Create a convolutional neural network
    # Create a convolutional neural network. learn 32 filters using a 3x3 kernel
    # each image is 28x28 pixels, one channel (grayscale)
    # Add a max-pooling layer, using 2x2 pool size
    # make input into one flat layer (vector)
    # Add a hidden layer with dropout
    # avoid overfitting!
    # Add an output layer with output units for all 10 digits. use softmax: probability distribution over the 10 digits
    # compile the model
    # fit the model


    # Create a convolutional neural network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=128)

    return model



if __name__ == '__main__':
    # Load data
    mnist, x_train, y_train, x_test, y_test = prepare_data()
    # Train neural network
    model = train_model(x_train, y_train)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")
