import tensorflow as tf
import tensorflow_datasets as tfds

keras = tf.keras

tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation
(rawTrain, rawValidation, rawTest), metadata = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    with_info=True,
    as_supervised=True
)

getLabelName = metadata.features["label"].int2str  # creates a function object that we can use to get labels

"""# display 2 images from the dataset
for image, label in rawTrain.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(getLabelName(label))
    # plt.show()"""

# Data preprocessing
IMG_SIZE = 160  # All images will be reshaped to 160x160


def formatExample(image, label):
    # returns an image that is reshaped to IMG_SIZE
    image - tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# Apply function to all our images using map
train = rawTrain.map(formatExample)
validation = rawValidation.map(formatExample)
test = rawTest.map(formatExample)

"""for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(getLabelName(label))"""

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

trainBatches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validationBatches = validation.batch(BATCH_SIZE)
testBatches = test.batch(BATCH_SIZE)

for img, label in rawTrain.take(2):
    print("Original shape:", img.shape)

for img, label in train.take(2):
    print("New shape:", img.shape)

# Picking pretrained model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
baseModel = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False,
                                              weights='imagenet')
baseModel.summary()

for image, _ in trainBatches.take(1):
    pass

featureBatch = baseModel(image)
print(featureBatch.shape)

# Freezing the base
baseModel.trainable = False
baseModel.summary()

# Adding our classifier
globalAverageLayer = tf.keras.layers.GlobalAveragePooling2D()
predictionLayer = keras.layers.Dense(1)
model = tf.keras.Sequential([
    baseModel,
    globalAverageLayer,
    predictionLayer
])
model.summary()

# Training the model
baseLearningRate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=baseLearningRate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# We can evaluate the model right now to see how it does before training it on our new images
initialEpochs = 3
validationSteps = 20

loss0, accuracy0 = model.evaluate(validationBatches, steps=validationSteps)

# Now we can train it on our images
history = model.fit(trainBatches,
                    epochs=initialEpochs,
                    validation_data=validationBatches)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
newModel = tf.keras.models.load_model('dogs_vs_cats.h5')
