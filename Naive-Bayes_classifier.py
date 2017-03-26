import random
import numpy as np
import pandas as pd
import math
import time
import sys

def main(argv):
    start = time.time()
    data, words = read_data()
    dev_data, test_data, train_data = split_data(data)

    frequencies_of_words = count_frequencies(train_data, words, argv[1])
    smoothed_frequencies = smooth_frequencies(frequencies_of_words)

    predicted_classes = predict_classes(smoothed_frequencies, dev_data)
    correct_count = count_correct_classes(predicted_classes, dev_data)
    print("Total dev sentences: %d"%len(dev_data))
    print("Correctly classified dev sentencec: %d"%correct_count)
    print("Runtime: %d"%(int)(time.time()-start))


# Make every sentence a tuple with the second element
# being the class label. (Obama or Trump)
# Read the data from the two input files
# Also make a complete list of all words that can be 
# used for smoothing purposes later
def read_data():
    with open('obama.txt') as f:
        obama_data = [(line, 'obama') for line in f]
    with open('trump.txt') as f:
        trump_data = [(line, 'trump') for line in f]

    words = list(set([ word for sentence in obama_data + trump_data\
                            for word in sentence[0].split()]))
    words.append('total_words')
    return [obama_data, trump_data], words


# Split the data into three parts: train, dev and test.
# Dev = 10%
# Test = 10%
# Train = 80%
# Shuffle the training data.
def split_data(data):
    length_obama = len(data[0])
    length_trump = len(data[1])
    dev_data = []
    test_data = []
    train_data = []

    for i in range(0, length_obama//10):
        dev_data.append(data[0][i])
    for i in range(0, length_trump//10):
        dev_data.append(data[1][i])
    random.shuffle(dev_data)

    for i in range(length_obama//10, length_obama//5):
        test_data.append(data[0][i])
    for i in range(length_trump//10, length_trump//5):
        test_data.append(data[1][i])
    random.shuffle(test_data)

    for i in range(length_obama//5, length_obama):
        train_data.append(data[0][i])
    for i in range(length_trump//5, length_trump):
        train_data.append(data[1][i])
    random.shuffle(train_data)

    return dev_data, test_data, train_data


# Count the frequencies of words in each class.
# return a 2D array-like with the counts of all words.
def count_frequencies(train_data, words, bool_train):
    if bool_train.lower() != "retrain" and bool_train.lower() != "yes":
        return pd.read_csv('counted_words_obama_trump.csv', index_col=0)

    frequency_table = pd.DataFrame(0, index=words, columns=['obama', 'trump'])
    for sentence in train_data:
        for word in sentence[0].split():
            frequency_table[sentence[1]][word] += 1

    frequency_table['obama']['total_words'] = sum(frequency_table['obama'])
    frequency_table['trump']['total_words'] = sum(frequency_table['trump'])

    frequency_table.to_csv("counted_words_obama_trump.csv")
    return frequency_table


# Use a smoothing method to smooth for zero counts.
def smooth_frequencies(frequencies_of_words):
    return frequencies_of_words


# Predict the classes of the sentences in the test data
# Return a list with the class labels
def predict_classes(frequencies_of_words, test_data):
    predicted_classes = []
    obama_prob = (frequencies_of_words['obama']['total_words'] / 
                sum(frequencies_of_words.loc['total_words']))
    trump_prob = (frequencies_of_words['trump']['total_words'] / 
                sum(frequencies_of_words.loc['total_words']))

    for sentence in test_data:
        obama_sentence_prob = 1.0
        trump_sentence_prob = 1.0
        for word in sentence[0].split():
            obama_sentence_prob *= ((frequencies_of_words['obama'][word]/
                                sum(frequencies_of_words.loc[word])) * 
                                obama_prob)
            trump_sentence_prob *= ((frequencies_of_words['trump'][word]/
                                sum(frequencies_of_words.loc[word])) * 
                                trump_prob)

        if (obama_sentence_prob >= trump_sentence_prob):
            predicted_classes.append('obama')
        else:
            predicted_classes.append('trump')
    return predicted_classes


# Count the number of correctly labeled sentences
# by looking them up in the compare data
def count_correct_classes(predicted_classes, correct_data):
    correct_count = 0
    for i in range(len(correct_data)):
        if predicted_classes[i] == correct_data[i][1]:
            correct_count += 1
    return correct_count


main(sys.argv)
