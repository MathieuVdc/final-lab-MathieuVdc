import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
import os

def __count_words_in_a_file(file_path):
    
    file = open(file_path, "r", encoding="utf-8")
    return len(Counter(file.read().split()))


def __list_words_in_a_file(file_path):
    
    file = open(file_path, "r", encoding="utf-8")
    return Counter(file.read().split())


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
        list_words = []
        for textfile in os.listdir(os.path.join(folder_path,directory)):
            list_words.extend(__list_words_in_a_file(os.path.join(folder_path,directory,textfile)))
        results[directory] = list_words
    return results

def most_popular_elements_in_list(list_to_top, number_wanted):

    return dict(Counter(list_to_top).most_common(number_wanted))

            
            
    