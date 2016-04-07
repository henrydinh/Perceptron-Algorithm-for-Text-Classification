# Henry Dinh
# CS 6375.001
# Assignment 3 - Perceptron training algorithm for text classification
# To test the program, read the README file for instructions

import os
import sys
import collections
import re
import copy

# Document class to store email instances
class Document:
    text = ""
    word_freqs = {}

    # spam or ham
    true_class = ""
    learned_class = ""

    # Constructor
    def __init__(self, text, counter, true_class):
        self.text = text
        self.word_freqs = counter
        self.true_class = true_class

    def getText(self):
        return self.text

    def getWordFreqs(self):
        return self.word_freqs

    def getTrueClass(self):
        return self.true_class

    def getLearnedClass(self):
        return self.learned_class

    def setLearnedClass(self, guess):
        self.learned_class = guess



# counts frequency of each word in the text files and order of sequence doesn't matter
def bagOfWords(text):
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)


# Read all text files in given directory and construct the data set
# the directory path should just be like "train/ham" for example
# storage is the dictionary to store the email in
# True class is the true classification of the email (spam or ham)
def makeDataSet(storage_dict, directory, true_class):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as text_file:
                # stores dictionary of dictionary of dictionary as explained above in the initialization
                text = text_file.read()
                storage_dict.update({dir_entry_path: Document(text, bagOfWords(text), true_class)})


# Set the stop words
def setStopWords(stop_word_text_file):
    stops = []
    with open(stop_word_text_file, 'r') as txt:
        stops = (txt.read().splitlines())
    return stops


# Remove stop words from data set and store in dictionary
def removeStopWords(stops, data_set):
    filtered_data_set = copy.deepcopy(data_set)
    for i in stops:
        for j in filtered_data_set:
            if i in filtered_data_set[j].getWordFreqs():
                del filtered_data_set[j].getWordFreqs()[i]
    return filtered_data_set


# Extracts the vocabulary of all the text in a data set
def extractVocab(data_set):
    v = []
    for i in data_set:
        for j in data_set[i].getWordFreqs():
            if j not in v:
                v.append(j)
    return v


# learns weights using the perceptron training rule
def learnWeights(weights, learning_constant, training_set, num_iterations, classes):
    # Adjust weights num_iterations times
    for i in num_iterations:
        # Go through all training instances and update weights
        for d in training_set:
            # Used to get the current perceptron's output. If > 0, then spam, else output ham.
            weight_sum = weights['weight_zero']
            for f in training_set[d].getWordFreqs():
                if f not in weights:
                    weights[f] = 0.0
                weight_sum += weights[f] * training_set[d].getWordFreqs()[f]
            perceptron_output = 0.0
            if weight_sum > 0:
                perceptron_output = 1.0
            target_value = 0.0
            if training_set[d].getTrueClass() == classes[1]:
                target_value = 1.0
            # Update all weights that are relevant to the instance at hand
            for w in training_set[d].getWordFreqs():
                weights[w] += float(learning_constant) * float((target_value - perceptron_output)) * \
                              float(training_set[d].getWordFreqs()[w])


# applies the algorithm to test accuracy on the test set. Returns the perceptron output
def apply(weights, classes, instance):
    weight_sum = weights['weight_zero']
    for i in instance.getWordFreqs():
        if i not in weights:
            weights[i] = 0.0
        weight_sum += weights[i] * instance.getWordFreqs()[i]
    if weight_sum > 0:
        # return is spam
        return 1
    else:
        # return is ham
        return 0


# Takes training directory containing spam and ham folder. Same with test directory
# Also takes number of iterations and learning rate as parameters
def main(train_dir, test_dir, iterations, learning_constant):
    # Create dictionaries and lists needed
    training_set = {}
    test_set = {}
    filtered_training_set = {}
    filtered_test_set = {}

    # Stop words to filter out
    stop_words = setStopWords('stop_words.txt')

    # ham = 0 for not spam, spam = 1 for is spam
    classes = ["ham", "spam"]

    # Number of iterations and learning constant (usually around .1 or .01)
    iterations = iterations
    learning_constant = learning_constant

    # Set up data sets. Dictionaries containing the text, word frequencies, and true/learned classifications
    makeDataSet(training_set, train_dir + "/spam", classes[1])
    makeDataSet(training_set, train_dir + "/ham", classes[0])
    makeDataSet(test_set, test_dir + "/spam", classes[1])
    makeDataSet(test_set, test_dir + "/ham", classes[0])

    # Set up data sets without stop words
    filtered_training_set = removeStopWords(stop_words, training_set)
    filtered_test_set = removeStopWords(stop_words, test_set)

    # Extract training set vocabulary
    training_set_vocab = extractVocab(training_set)
    filtered_training_set_vocab = extractVocab(filtered_training_set)

    # store weights as dictionary. w0 initiall 1.0, others initially 1.0. token : weight value
    weights = {'weight_zero': 1.0}
    filtered_weights = {'weight_zero': 1.0}
    for i in training_set_vocab:
        weights[i] = 0.0
    for i in filtered_training_set_vocab:
        filtered_weights[i] = 0.0

    # Learn weights using the training_set and the filtered_training_set
    learnWeights(weights, learning_constant, training_set, iterations, classes)
    learnWeights(filtered_weights, learning_constant, filtered_training_set, iterations, classes)

    #Apply the algorithm on the test set and report accuracy
    num_correct_guesses = 0
    for i in test_set:
        guess = apply(weights, classes, test_set[i])
        if guess == 1:
            test_set[i].setLearnedClass(classes[1])
            if test_set[i].getTrueClass() == test_set[i].getLearnedClass():
                num_correct_guesses += 1
        if guess == 0:
            test_set[i].setLearnedClass(classes[0])
            if test_set[i].getTrueClass() == test_set[i].getLearnedClass():
                num_correct_guesses += 1

    # Apply algorithm again on test set without any stop words and report accuracy
    filt_num_correct_guesses = 0
    for i in filtered_test_set:
        guess = apply(filtered_weights, classes, filtered_test_set[i])
        if guess == 1:
            filtered_test_set[i].setLearnedClass(classes[1])
            if filtered_test_set[i].getTrueClass() == filtered_test_set[i].getLearnedClass():
                filt_num_correct_guesses += 1
        if guess == 0:
            filtered_test_set[i].setLearnedClass(classes[0])
            if filtered_test_set[i].getTrueClass() == filtered_test_set[i].getLearnedClass():
                filt_num_correct_guesses += 1

    # Report accuracy
    print "Learning constant: %.4f" % float(learning_constant)
    print "Number of iterations: %d" % int(iterations)
    print "Emails classified correctly: %d/%d" % (num_correct_guesses, len(test_set))
    print "Accuracy: %.4f%%" % (float(num_correct_guesses) / float(len(test_set)) * 100.0)
    print "Filtered emails classified correctly: %d/%d" % (filt_num_correct_guesses, len(filtered_test_set))
    print "Filtered accuracy: %.4f%%" % (float(filt_num_correct_guesses) / float(len(filtered_test_set)) * 100.0)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])