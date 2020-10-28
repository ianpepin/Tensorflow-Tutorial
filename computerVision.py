import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# Load and split dataset
(trainImages, trainLabels), (testImages, testLabels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
trainImages, testImages = trainImages / 255.0, testImages / 255.0

classNames = ["airplane", "automobile", "bird", "car", "deer", "dog", "frog", "horse", "ship", "truck"]
"""
# Let's look at one image
IMG_INDEX = 1       # change this to look at other images

"""
# plt.imshow(trainImages[IMG_INDEX], cmap=plt.cm.binary)
# plt.xlabel(classNames[trainLabels[IMG_INDEX][0]])
# plt.show()


# Convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.summary()

# Adding dense layers (classifier)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))     # 10: amount of classes we have (final output)
model.summary()

# Training
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
history = model.fit(trainImages, trainLabels, epochs=10,
                    validation_data=(testImages, testLabels))

# Evaluating the model
testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=2)
print(testAcc)


"""Data augmentation"""


# creates a data generator object that transforms images
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# pick an image to transform
testImg = trainImages[14]
img = image.img_to_array(testImg)       # convert image to numpy array
img = img.reshape((1,) + img.shape)     # reshape image

i = 0
for batch in datagen.flow(img, save_prefix="test", save_format="jpeg"):     # this loops forever until we break, saving images to current directory
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 3:       # show 4 images
        break
plt.show()
