import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import os
import string
from sklearn import preprocessing



def __count_words_in_a_file(file_path):
    
    file = open(file_path, "r", encoding="utf-8")
    return len(Counter(file.read().split()))


def count_average_number_of_words_per_class(folder_path):
    
    results = {}
    for directory in os.listdir(folder_path):
        list_number_words = []
        for textfile in os.listdir(os.path.join(folder_path,directory)):
            list_number_words.append(__count_words_in_a_file(os.path.join(folder_path,directory,textfile)))
        results[directory] = np.mean(list_number_words)
    return results


def __get_lines_cleaned(file_path):
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


def data_pre_processing(input_csv_path):
   
    raw_data = pd.read_csv(input_csv_path, sep = ",")
    x = []
    y = []
    for index, row in raw_data.iterrows():
        ocr_path = row['img_path'].replace('.jpg','.txt')
        x.append(__get_lines_cleaned(os.path.join('tobacco-lab/data/Tobacco3482-OCR/'+ ocr_path)))
        y.append(row['label'])
    labelencod = preprocessing.LabelEncoder()
    y_num = list(labelencod.fit_transform(y))
    return [x,y]


    

            
    