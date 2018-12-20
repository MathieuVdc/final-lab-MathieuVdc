import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import os
import string

def __count_words_in_a_file(file_path):
    
    file = open(file_path, "r", encoding="utf-8")
    return len(Counter(file.read().split()))


def __get_lines(file_path):
    lines = open(file_path).readlines()
    newlines = []
    table = str.maketrans('', '', string.punctuation+'’°“‘”—»«®©℠™')
    for line in lines :
        clean = line.rstrip()
        clean_better = (clean.translate(table)).lower()
        clean_better_better = clean_better.strip()
        if (clean_better_better != ''):
            newlines.append(clean_better_better)
    return newlines


def __list_words_in_a_file(file_path):
    
    lines = __get_lines(file_path)
    count_vec = CountVectorizer()
    vector = count_vec.fit_transform(lines)
    return count_vec.get_feature_names()


def count_average_number_of_words_per_class(folder_path):
    
    results = {}
    for directory in os.listdir(folder_path):
        list_number_words = []
        for textfile in os.listdir(os.path.join(folder_path,directory)):
            list_number_words.append(__count_words_in_a_file(os.path.join(folder_path,directory,textfile)))
        results[directory] = np.mean(list_number_words)
    return results


def list_of_words_per_class(folder_path):
    
    results = {}
    for directory in os.listdir(folder_path):
        print(directory)
        list_words = []
        for textfile in os.listdir(os.path.join(folder_path,directory)):
            print(textfile)
            list_words.extend(__list_words_in_a_file(os.path.join(folder_path,directory,textfile)))
        results[directory] = list_words
    return results


def most_popular_elements_in_list(list_to_top, number_wanted):

    return dict(Counter(list_to_top).most_common(number_wanted))

            
            
    