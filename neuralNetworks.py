import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Data set
fashion_mnist = keras.datasets.fashion_mnist  # load dataset
(trainImages, trainLabels), (testImages, testLabels) = fashion_mnist.load_data()  # split into testing and training
print(trainImages.shape)  # 60000 images, 28x28 pixels
print(trainImages[0, 23, 23])  # one pixel (0 is black, 255 is white)

print(trainLabels[:10])  # first 10 training labels

classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", 'Bag", "Ankle boot']

"""
plt.figure()
plt.imshow(trainImages[3])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# Data preprocessing
trainImages = trainImages / 255.0
testImages = testImages / 255.0

# Build the model
model = keras.Sequential([  # Most basic neural network: passes data sequentially (left to right)
    keras.layers.Flatten(input_shape=(28, 28)),
    # input layer       -> Flatten: takes shape and flatten all pixels into 784 pixels
    keras.layers.Dense(128, activation="relu"),
    # hidden layer      -> Dense: every node in previous layer is connected to nodes in this layer
    keras.layers.Dense(10, activation="softmax")  # output layer: has as many neurons as classes
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(trainImages, trainLabels, epochs=5)

# Evaluating the model
testLoss, testAccuracy = model.evaluate(testImages, testLabels, verbose=1)
print("Test accuracy: ", testAccuracy)

# Making predictions
predictions = model.predict(testImages)
print(predictions[1])
print(np.argmax(predictions[1]))  # returns the index of the maximum value from a numpy array
print(classNames[np.argmax(predictions[1])])

"""plt.figure()
plt.imshow(testImages[1])
plt.colorbar()
plt.grid(False)
plt.show()"""

# Verifying predictions
COLOR = "white"
plt.rcParams["text.color"] = COLOR
plt.rcParams["axes.labelcolor"] = COLOR


def predict(model, image, correctLabel):
    classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    prediction = model.predict(np.array([image]))
    predictedClass = classNames[np.argmax(prediction)]
    showImage(image, classNames[correctLabel], predictedClass)


def showImage(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def getNumber():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
        if 0 <= num <= 100:
            return int(num)
        else:
            print("Try again...")


num = getNumber()
image = testImages[num]
label = testLabels[num]
predict(model, image, label)
