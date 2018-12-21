import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd
import os
import string
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")



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


def __get_text_cleaned(file_path):
    lines = open(file_path).readlines()
    newlines = []
    table = str.maketrans('', '', string.punctuation+'’°“‘”—»«®©℠™')
    for line in lines :
        clean = line.rstrip()
        clean_better = (clean.translate(table)).lower()
        clean_better_better = clean_better.strip()
        if (clean_better_better != ''):
            newlines.append(clean_better_better)
    return " ".join(newlines)


def __data_organizing(input_csv_path):
   
    raw_data = pd.read_csv(input_csv_path, sep = ",")
    x = []
    y = []
    for index, row in raw_data.iterrows():
        ocr_path = row['img_path'].replace('.jpg','.txt')
        x.append(__get_text_cleaned(os.path.join('tobacco-lab/data/Tobacco3482-OCR/'+ ocr_path)))
        y.append(row['label'])
    labelencod = preprocessing.LabelEncoder()
    y_num = list(labelencod.fit_transform(y))
    print("taille de X : "+str(len(x)))
    print("taille de y : "+str(len(x)))
    return [x,y_num]

def __data_splitting(x,y):
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state=42)
    print("taille de X_train : "+str(len(X_train)))
    print("taille de X_val : "+str(len(X_val)))
    print("taille de X_test : "+str(len(X_test)))
    print("taille de y_train : "+str(len(y_train)))
    print("taille de y_val : "+str(len(y_val)))
    print("taille de y_test : "+str(len(y_test)))
    return [X_train, X_val, X_test, y_train, y_val, y_test]


def organize_and_split_data(input_csv_path):
    
    listXy = __data_organizing(input_csv_path)
    datasets = __data_splitting(listXy[0], listXy[1])
    return datasets



def __tokenizing(X_train, X_val, X_test, max_feat = None):
    
    print("Tokenization...")
    vectorizer = CountVectorizer(max_features=max_feat)
    vectorizer.fit(X_train)
    X_train_cv = vectorizer.transform(X_train)
    X_val_cv = vectorizer.transform(X_val)
    X_test_cv = vectorizer.transform(X_test)
    print("Terminé !")
    print("Taille du vocabulaire :")
    print(str(len(vectorizer.get_feature_names())))
    return [X_train_cv, X_val_cv, X_test_cv]


def __tfidf(X_train_cv, X_val_cv, X_test_cv):
    
    print("TFIDF...")
    tf_transformer = TfidfTransformer().fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_val_tf = tf_transformer.transform(X_val_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)
    print("Terminé !")
    return [X_train_tf, X_val_tf, X_test_tf]


def tokenizing_and_tfidf(X_train, X_val, X_test, max_feat = None):
    
    cv = __tokenizing(X_train, X_val, X_test, max_feat)
    return __tfidf(cv[0], cv[1], cv[2])
 
    
def my_MultinomialNB(X_train, X_val, X_test, y_train, y_val, y_test, a = 1.0, cross_val = 5):
    
    clf = MultinomialNB(alpha=a)
    print("Entraînement...")
    clf.fit(X_train,y_train)
    print("Terminé !")
    print("Evaluation par Cross-Validation ("+str(cross_val)+") avec alpha = "+str(a)+" :")
    cross_validation_score = np.mean(cross_val_score(clf, X_val, y_val, cv=cross_val))
    print(cross_validation_score)
    print("Précision/Score sur les données de Test : ")
    score = clf.score(X_test,y_test)
    print(score)  
    y_pred = clf.predict(X_test)
    print("")    
    print("Matrice de Confusion :")
    print(confusion_matrix(y_pred,y_test))
    print("")
    print("Rapport de classification :")
    print(classification_report(y_pred,y_test))
    return [clf, a, cross_validation_score, score]


def my_MultinomialNB_mute(X_train, X_val, X_test, y_train, y_val, y_test, a = 1.0, cross_val = 5):
    
    clf = MultinomialNB(alpha=a)
    clf.fit(X_train,y_train)
    cross_validation_score = np.mean(cross_val_score(clf, X_val, y_val, cv=cross_val))
    score = clf.score(X_test,y_test)
    y_pred = clf.predict(X_test)
    return [clf, a, cross_validation_score, score]

    


            
    