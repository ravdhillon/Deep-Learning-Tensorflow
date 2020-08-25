'''
create the lexicon for the data words.
'''
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
import os
from collections import Counter

lemmatizer = WordNetLemmatizer()
lines = 1000000

dataset_dir = os.path.join(os.getcwd(), 'Dataset')
positive_dataset_file = os.path.join(dataset_dir, 'pos.txt')
negative_dataset_file = os.path.join(dataset_dir, 'neg.txt')

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    # Lemmatize Words
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    word_count = Counter(lexicon)
    # word_count = {'the': 545454, 'and':434343} something like this. Can I use MapReduce ?
    l2 = []
    for w in word_count:
        # this is to filter the common words and also really rare words we don't care about. We can play with these numbers.
        # The lexicon should not be huge
        if 1000 > word_count[w] > 50:
            l2.append(w)

    return l2

# Using Lexicon to classify feature_set
def sample_handling(sample, lexicon, classification):
    feature_set = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            feature_set.append([features, classification])
    return feature_set


def create_feature_set_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)

    features = []
    features += sample_handling(pos, lexicon, [1,0]) #[1,0] are the classes
    features += sample_handling(neg, lexicon, [0,1])

    features = np.array(features)

    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size]) # this is all 0th element and take all of these elements till test size
    train_y = list(features[:, 1][:-testing_size]) # this is all 0th element and take all of these elements till test size

    test_x = list(features[:, 0][-testing_size:]) # this is all 0th element and take all of these elements till test size
    test_y = list(features[:, 1][-testing_size:]) # this is all 0th element and take all of these elements till test size

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_set_and_labels(positive_dataset_file, negative_dataset_file)
    with open('sentiment-pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)