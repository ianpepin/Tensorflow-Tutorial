"""
Represents each word or character in a sentence as a unique integer and maintaining the order of these words. Has a few
issues with it. Ideally when we encode words, we would like similar words to have similar labels and different words to have
very different labels. For example, the words happy and joyful should probably have very similar labels so we can determine
that they are similar. While words like horrible and amazing should probably have very different labels. The method we looked
at above won't be able to do something like this for us. This could mean that the model will have a very difficult time determining
if two words are similar or not which could result in some pretty drastic performance impacts.
"""

vocab = {}
wordEncoding = 1


def oneHotEncoding(text):
    global wordEncoding

    words = text.lower().split(" ")
    encoding = []

    for word in words:
        if word in vocab:
            code = vocab[word]
            encoding.append(code)
        else:
            vocab[word] = wordEncoding
            encoding.append(wordEncoding),
            wordEncoding += 1
    return encoding


text = "this is a test to see if this test will work is is test a a"
encoding = oneHotEncoding(text)
print(encoding)
print(vocab)

positiveReview = "I thought the movie was going to be bad but it was actually amazing"
negativeReview = "I thought the movie was going to be amazing but it was actually bad"

posEncode = oneHotEncoding(positiveReview)
negEncode = oneHotEncoding(negativeReview)

print("Positive:", posEncode)
print("Negative:", negEncode)
