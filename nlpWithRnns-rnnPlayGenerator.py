import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
import numpy as np
import os

pathToFile = tf.keras.utils.get_file("shakespeare.txt",
                                     "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

# Read then decode for py2 compat.
text = open(pathToFile, "rb").read().decode(encoding="utf-8")
# length of text is the number of characters in it
print("Length of text: {} characters".format(len(text)))

# first 250 characters in text
print(text[:250])

# Encoding
vocab = sorted(set(text))  # sorts all unique characters in text
# creating a mapping from unique characters to indices
charToIndex = {u: i for i, u in enumerate(vocab)}
indexToChar = np.array(vocab)


def textToInt(text):
    return np.array([charToIndex[c] for c in text])


textAsInt = textToInt(text)


# print(textAsInt)

# let's look at how part of our text is encoded
# print("Text: ", text[:13])
# print("Encoded: ", textToInt(text[:13]))  # eah integer represents a character


def intToText(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return "".join(indexToChar[ints])


# print(intToText(textAsInt[:13]))

# Creating training examples
seqLength = 100  # length of sequence for a training example
examplesPerEpoch = len(text) // (seqLength + 1)

# Create training examples / targets
charDataset = tf.data.Dataset.from_tensor_slices(textAsInt)

sequences = charDataset.batch(seqLength + 1, drop_remainder=True)


# split sequences of length 101 into input and output
def splitInputTarget(chunk):  # for example: hello
    inputText = chunk[:-1]  # hell
    targetText = chunk[1:]  # ello
    return inputText, targetText  # hell, ello


dataset = sequences.map(splitInputTarget)  # we use map to apply the above function to every entry

for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(intToText(x))
    print("\nOUTPUT")
    print(intToText(y))

# make training batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# Building the model
def buildModel(vocabSize, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabSize, embeddingDim, batch_input_shape=[batchSize, None]),
        # length of the batch is unknown (None)
        tf.keras.layers.LSTM(rnnUnits, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        tf.keras.layers.Dense(vocabSize)
    ])
    return model


model = buildModel(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

# Creating a loss function
for inputExampleBatch, targetExampleBatch in data.take(1):
    exampleBatchPredictions = model(
        inputExampleBatch)  # ask our model for a prediction on our first batch of training data (64 entries)
    print(exampleBatchPredictions.shape, "# (batchSize, sequenceLength, vocabSize)")  # print out the output shape

# we can see that the prediction is an array of 64 arrays, one for each entry in the batch
print(len(exampleBatchPredictions))
print(exampleBatchPredictions)

# let's examine one prediction
pred = exampleBatchPredictions[0]
print(len(pred))
print(pred)
# notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step

# and finally well look at a prediction at the first timestep
timePred = pred[0]
print(len(timePred))
print(timePred)
# and of course its 65 values representing the probabillity of each character occuring next

# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
sampledIndices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampledIndices = np.reshape(sampledIndices, (1, -1))[0]
predictedChars = intToText(sampledIndices)

predictedChars  # and this is what the model predicted for training sequence 1


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# Compiling the model
model.compile(optimizer='adam', loss=loss)

# Creating checkpoints
# Directory where the checkpoints will be saved
checkpointDir = './training_checkpoints'
# Name of the checkpoint files
checkpointPrefix = os.path.join(checkpointDir, "ckpt_{epoch}")

checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix,
    save_weights_only=True)

# Training
history = model.fit(data, epochs=50, callbacks=[checkpointCallback])

# Loading the model
model = buildModel(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpointDir))
model.build(tf.TensorShape([1, None]))
checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))


# Generating text
def generateText(model, startString):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    numGenerate = 800

    # Converting our start string to numbers (vectorizing)
    inputEval = [charToIndex[s] for s in startString]
    inputEval = tf.expand_dims(inputEval, 0)

    # Empty string to store our results
    textGenerated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(numGenerate):
        predictions = model(inputEval)
        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predictedId = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predictedId], 0)

        textGenerated.append(indexToChar[predictedId])

    return startString + ''.join(textGenerated)


inp = input("Type a starting string: ")
print(generateText(model, inp))
