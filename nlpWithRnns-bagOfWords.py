# Bag of words
vocab = {}      # maps word to integer representing it
wordEncoding = 1


def bagOfWords(text):
    global wordEncoding

    words = text.lower().split(" ")     # create a list of all of the words in the text, we'll assume there is no grammar in our text for this example
    bag = {}        # stores all of the encodings and their frequency

    for word in words:
        if word in vocab:
            encoding = vocab[word]      # get encoding from vocab
        else:
            vocab[word] = wordEncoding
            encoding = wordEncoding
            wordEncoding += 1

        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
    return bag


text = "this is a test to see if this test will work is is test a a"
bag = bagOfWords(text)
print(bag)
print(vocab)

positiveReview = "I thought the movie was going to be bad but it was actually amazing"
negativeReview = "I thought the movie was going to be amazing but it was actually bad"

posBag = bagOfWords(positiveReview)
negBag = bagOfWords(negativeReview)

print("Positive: ", posBag)
print("Negative: ", negBag)
