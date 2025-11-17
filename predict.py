import tensorflow as tf
import numpy as np 
import cv2

""""
model = tf.keras.models.load_model("D:\Personal_Dir\mnist_project\mnist_cnn_model.h5")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

index = 0
img = x_test[index]

img_norm = img.astype("float32") / 255.0
img_norm = np.expand_dims(img_norm, axis=-1)
img_norm = np.expand_dims(img_norm, axis=0)

pred = model.predict(img_norm)
digit = np.argmax(pred)

print("Actual Label:", y_test[index])
print("Predicted Digit:", digit)

"""

def preprocessing_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28,28))
    img_norm = img.astype("float32") / 255.0

    img_norm = np.expand_dims(img_norm, axis=-1)

    img_norm = np.expand_dims(img_norm, axis = 0)

    return img_norm

model = tf.keras.models.load_model("D:\Personal_Dir\mnist_project\mnist_cnn_model.h5")

img_path = "D:\Personal_Dir\mnist_project\digit.png"

img = preprocessing_image(img_path)

pred = model.predict(img)

digit = np.argmax(pred)

print("Predicted digit:", digit)