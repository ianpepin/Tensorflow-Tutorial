import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]

trainPath = tf.keras.utils.get_file("iris_training.csv",
                                    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
testPath = tf.keras.utils.get_file("iris_test.csv",
                                   "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(trainPath, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(testPath, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras to grab our datasets and read them into a pandas dataframe

# print(train.head())

# Pop the species column off and use it as our label
trainY = train.pop("Species")
testY = test.pop("Species")
# print(train.head())  # The species column is now gone
# print(train.shape)  # We have 120 entries with 4 features


# Input function
def inputFn(features, labels, training=True, batchSize=256):
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batchSize)


# Feature columns: describe how to use the input
myFeatureColumns = []
for key in train.keys():  # train.keys(): gives us all the columns
    myFeatureColumns.append(tf.feature_column.numeric_column(key=key))
print(myFeatureColumns)

# Building the model
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns=myFeatureColumns,
    # Two hidden layers of 30 and 10 nodes respectively
    hidden_units=[30, 10],
    # The model must choose between 3 classes
    n_classes=3)

# Training the model
classifier.train(input_fn=lambda: inputFn(train, trainY, training=True), steps=10000)
# We include a lambda to avoid creating an inner function previously (calls function that returns inputFn)

evalResult = classifier.evaluate(input_fn=lambda: inputFn(test, testY, training=False))
print("\nTest set accuracy: {accuracy:0.3f}\n".format(**evalResult))


# Predictions
def inputFunction(features, batch_size=256):
    # Convert the inputs to a Dataset without labels because we don't know the labels when making predictions.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit():
            valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: inputFunction(predict))
for predDict in predictions:
    classId = predDict['class_ids'][0]
    probability = predDict['probabilities'][classId]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[classId], 100 * probability))
