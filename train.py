import argparse
import numpy as np
import tensorflow as tf

def load_and_inspect(subset_size = 1000, test_subset = 1000):
    #here im loading the full dataset from MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("Full dataset shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #for my laptop processing speed, I've took first 1000 for each and test 
    x_train_sub = x_train[:subset_size]
    y_train_sub = y_train[:subset_size]
    x_test_sub = x_test[:test_subset]
    y_test_sub = y_test[:test_subset]

    print(f"Using subset -> train: {x_train_sub.shape}, test: {x_test_sub.shape}")

    #datatype are uint8 min-0(black) max-255(white)
    print("dtype:", x_train_sub.dtype, "min/max (train):", x_train_sub.min(), x_train_sub.max())

    unique, counts = np.unique(y_train_sub, return_counts=True)

    print("Train lables distribution (Value:count):")

    for v,c in zip(unique,counts):
        print(f" {v} : {c}")
    return (x_train_sub, y_train_sub), (x_test_sub, y_test_sub) 

def main(subset_size = 1000, test_subset = 1000):
    (x_train, y_train), (x_test, y_test) = load_and_inspect(
    subset_size=subset_size,test_subset=test_subset)   
    print("\nSTEP 1 completed: Data loaded and inspected. If class counts look reasonable, proceed to the next step.")

    #step 2: preprocess (normalize, reshape, one-hot)
    print("\nSTEP 2: Preprocessing...")

    # Normalize only X
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape for CNN (add channel dimension)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("After reshape:", x_train.shape, x_test.shape)

    # One-hot encode labels (do NOT divide labels)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    print("After one-hot:", y_train.shape, y_test.shape)


    #step 3: Build the model
    print("\nSTEP 3: Building the model...")

    def build_model():
        model = tf.keras.models.Sequential([
            
            # 1st Convolution Block
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),

            # 2nd Convolution Block
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),

            # Flatten + Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    print("Model built and compiled successfully.")

    model = build_model()

    history = model.fit(
        x_train, y_train,
        epochs = 10,
        batch_size = 32,
        validation_data = (x_test, y_test)
    )
    model.save("mnist_cnn_model.h5")
    print("Model saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py - stepwise MNIST training (STEP 1: Load & Inspect)")
    parser.add_argument("--subset", type=int, default=1000, help="Number of training examples to use (default 1000)")
    parser.add_argument("--test_subset", type=int, default=1000, help="Number of test examples to use (default 1000)")
    args = parser.parse_args()
    main(subset_size=args.subset, test_subset=args.test_subset)


