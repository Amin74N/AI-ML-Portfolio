# Warning Prevention:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ----------------------------------------------------------------------------------------------------------------------
# Imports:
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Dataset:
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Dataset Shape (How many images with how many pixels for each?):
print(train_images.shape)
# A specific Pixel of The First Picture (0 means BLACK & 1 means WHITE):
print(train_images[0,23,23])
# Dataset Labels (Integers between 0-9 representing a specific article of clothing):
print(train_labels[:10])
# Dataset Class Names (Creating an array of label names to indicate which number is which article of clothing):
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# ----------------------------------------------------------------------------------------------------------------------
# Looking at an image:
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Data Preprocessing:
train_images = train_images / 255.0
test_images = test_images / 255.0
# ----------------------------------------------------------------------------------------------------------------------
# Model Building:
model = keras.Sequential([keras.layers.Input(shape= (28,28)),
                          keras.layers.Flatten(),                         # 1. Input  Layer
                          keras.layers.Dense(128, activation= 'relu'),    # 2. Hidden Layer
                          keras.layers.Dense(10, activation= 'softmax')]) # 3. Output Layer
# ----------------------------------------------------------------------------------------------------------------------
# Model Compiling:
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
# ----------------------------------------------------------------------------------------------------------------------
# Model Training:
model.fit(train_images, train_labels, epochs= 1)
# ----------------------------------------------------------------------------------------------------------------------
# Model Evaluation:
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose= 1)
print("Test accuracy:", test_acc)
# ----------------------------------------------------------------------------------------------------------------------
# Making predictions:
predictions = model.predict(test_images)
print("The image is a ", class_names[np.argmax(predictions[2])])
plt.figure()
plt.imshow(test_images[2])
plt.colorbar()
plt.grid(False)
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Verifying predictions:
def predict(model, image, correct_label):
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)
def show_image(image, label, guess):
    plt.figure()
    plt.imshow(image)
    plt.title("Expected:" + label)
    plt.xlabel("Guess:" + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()
def get_number():
    while True:
        num = input("Enter a number: ")
        if num.isdigit():
            num = int(num)
            if num >= 0 and num <= 1000:
                return num
        else:
            print("Try Again ...")
num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
