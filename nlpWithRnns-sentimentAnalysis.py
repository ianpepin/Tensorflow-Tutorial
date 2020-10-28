import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words=VOCAB_SIZE)

print(trainData[0])

# Adjust length of review to max length of 250 words
trainData = sequence.pad_sequences(trainData, MAXLEN)
testData = sequence.pad_sequences(testData, MAXLEN)

# Creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")  # >0.5: positive, <0.5, negative
])
model.summary()

# Training
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(trainData, trainLabels, epochs=10, validation_split=0.2)

results = model.evaluate(testData, testLabels)
print(results)

# Making predictions
wordIndex = imdb.get_word_index()


def encodeText(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)  # convert text into tokens (individual words)
    tokens = [wordIndex[word] if word in wordIndex else 0 for word in
              tokens]  # if word in tokens is in mapping, replace its location in the list with that integer, otherwise 0
    return sequence.pad_sequences([tokens], MAXLEN)[0]


text = "that movie was just amazing, so amazing"
encoded = encodeText(text)
print(encoded)

# Decode function
reverseWordIndex = {value: key for (key, value) in wordIndex.items()}


def decodeIntegers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverseWordIndex[num] + " "
    return text[:-1]


print(decodeIntegers(encoded))


# Time to make a prediction
def predict(text):
    encodedText = encodeText(text)
    pred = np.zeros((1, 250))
    pred[0] = encodedText
    result = model.predict(pred)
    return result[0]


positiveReview = "That movie was! really loved it and would great watch it again because it was amazingly great"
print(predict(positiveReview))

negativeReview = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
print(predict(negativeReview))
