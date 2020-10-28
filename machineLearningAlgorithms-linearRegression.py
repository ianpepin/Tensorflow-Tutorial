from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

tf.keras.backend.set_floatx('float64')

# Load dataset
# df: dataframe
dfTrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")  # training data
dfEval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")  # testing data
print(dfTrain.head())  # first 5 entries in dataset
yTrain = dfTrain.pop("survived")
yEval = dfEval.pop("survived")
# print(dfTrain.describe())
# print(dfTrain.shape)
print(dfTrain.head())
print(dfTrain.loc[0], yTrain.loc[0])  # prints specific row

# Categorical: something that is a category
CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
NUMERIC_COLUMNS = ["age", "fare"]

# featureColumns: what we need to fee to our linear model to make productions
featureColumns = []
for featureName in CATEGORICAL_COLUMNS:
    vocabulary = dfTrain[featureName].unique()  # gets a list of all unique values from given feature column
    featureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list(featureName, vocabulary))

for featureName in NUMERIC_COLUMNS:
    featureColumns.append(tf.feature_column.numeric_column(featureName, dtype=tf.float32))

print(featureColumns)


# Input function
def makeInputFn(dataDf, labelDf, numEpochs=10, shuffle=True, batchSize=32):
    def inputFunction():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(dataDf), labelDf))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batchSize).repeat(numEpochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset
    return inputFunction  # return a function object for use


trainInputFn = makeInputFn(dfTrain, yTrain)  # here we will call  the inputFunction that was returned to us to get a dataset object we can feed to the model
evalInputFn = makeInputFn(dfEval, yEval, numEpochs=1, shuffle=False)

# Creating the model
linearEst = tf.estimator.LinearClassifier(feature_columns=featureColumns)  # We create a linear estimator by passing the feature columns we created earlier

# Training the model
linearEst.train(trainInputFn)  # train
result = linearEst.evaluate(evalInputFn)    # get model metrics/stats by testing on testing data
clear_output()  # clears console output
print(result["accuracy"])   # the result variable is simply a dict of stats about our model

# Get predictions from model
result = list(linearEst.predict(evalInputFn))
print(dfEval.loc[0])   # description of a person
print(yEval.loc[0])    # prints if that person survived or not
print(result[0]["probabilities"][1])   # chances of survival: 1, chances of not surviving: 0

# Graphs
# dfTrain.age.hist(bins=20)  # age
# dfTrain.sex.value_counts().plot(kind="barh")  # sex
# dfTrain["class"].value_counts().plot(kind="barh")  # class
# pd.concat([dfTrain, yTrain], axis=1).groupby("sex").survived.mean().plot(kind="barh").set_xlabel("% survive")  # % survival by sex
# plt.show()
