#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:31:00 2018
@author: Mathieu VANDECASTEELE
"""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
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
import logging
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")


def __count_words_in_a_file(file_path):
    """
        Compte le nombre de mots dans un fichier txt.
    """
    file = open(file_path, "r", encoding="utf-8")
    return len(Counter(file.read().split()))


def count_average_number_of_words_per_class(folder_path):
    """
        compte le nombre moyen de mots par classe.
    """
    results = {}
    for directory in os.listdir(folder_path):
        list_number_words = []
        for textfile in os.listdir(os.path.join(folder_path, directory)):
            list_number_words.append(
                __count_words_in_a_file(
                    os.path.join(
                        folder_path,
                        directory,
                        textfile)))
        results[directory] = np.mean(list_number_words)
    return results


def __get_text_cleaned(file_path):
    """
       À partir d'un fichier txt en entré, retourne un texte des lignes
       de ce dernier entièrement nettoyé dans une liste.
    """
    lines = open(file_path).readlines()
    newlines = []
    table = str.maketrans('', '', string.punctuation + '’°“‘”—»«®©℠™')
    for line in lines:
        clean = line.rstrip()
        clean_better = (clean.translate(table)).lower()
        clean_better_better = clean_better.strip()
        if (clean_better_better != ''):
            newlines.append(clean_better_better)
    return " ".join(newlines)


def __data_organizing(input_csv_path):
    """
        À partir du .csv, crée la paire principale X et y pour constituant
        le set principal de données.
    """
    raw_data = pd.read_csv(input_csv_path, sep=",")
    x = []
    y = []
    for index, row in raw_data.iterrows():
        ocr_path = row['img_path'].replace('.jpg', '.txt')
        x.append(
            __get_text_cleaned(
                os.path.join(
                    'data/Tobacco3482-OCR/' +
                    ocr_path)))
        y.append(row['label'])
    labelencod = preprocessing.LabelEncoder()
    y_num = list(labelencod.fit_transform(y))
    print("taille de X : " + str(len(x)))
    print("taille de y : " + str(len(x)))
    return [x, y_num]


def __data_splitting(x, y):
    """
        Renvoie 4 jeux à partir d'un jeu de données principal.
        x est un vecteur contenant des listes, chaque liste contient le texte
        d'un document. y est les labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)
    X_trainval = X_train + X_val
    y_trainval = y_train + y_val
    print("Fabriqué 4 jeux de données depuis les données : X_train, X_val, X_test, X_trainval (X_train+X_val) ")
    print("taille de train : " + str(len(X_train)))
    print("taille de val : " + str(len(X_val)))
    print("taille de trainval : " + str(len(X_trainval)))
    print("taille de test : " + str(len(X_test)))
    return [
        X_train,
        X_val,
        X_test,
        X_trainval,
        y_train,
        y_val,
        y_test,
        y_trainval]


def organize_and_split_data(input_csv_path):
    """
        Concatène les deux opérations précédentes.
    """
    listXy = __data_organizing(input_csv_path)
    datasets = __data_splitting(listXy[0], listXy[1])
    return datasets


def tokenizing(set_to_fit, sets_to_transform, max_feat=None, mdf=1.0):
    """
        Réalise un Bag of Words. set_to_fit est un set de données
        utilisé pour la fonction fit et sets_to_transforme est une liste
        des sets qui se verront appliqués la fonction transform en dehors
        du set_to_fit.
    """
    print("Tokenization...")
    vectorizer = CountVectorizer(max_features=max_feat, max_df=mdf)
    vectorizer.fit(set_to_fit)
    sets_cv = []
    sets_cv.append(vectorizer.transform(set_to_fit))

    if (isinstance(sets_to_transform[0], list)):
        for dataset in sets_to_transform:
            sets_cv.append(vectorizer.transform(dataset))
    else:
        sets_cv.append(vectorizer.transform(sets_to_transform))

    print("Terminé !\n")
    print("Taille du vocabulaire :")
    print(str(len(vectorizer.get_feature_names())))
    print('')
    print('Nombre de sets retournés : ' + str(len(sets_cv)))
    return sets_cv


def __tfidf(set_cv_to_fit, sets_cv_to_transform):
    """
        Réalise la TF-IDF d'un Bag of Words.
        set_cv_to_fit est un set de données
        utilisé pour la fonction fit et sets_cv_to_transforme
        est une liste des sets qui se verront appliqués la fonction
        transform en dehors du set_cv_to_fit.
    """
    print("TFIDF...")
    tf_transformer = TfidfTransformer().fit(set_cv_to_fit)
    sets_tf = []
    sets_tf.append(tf_transformer.transform(set_cv_to_fit))

    if (isinstance(sets_cv_to_transform[0], list)):
        for dataset in sets_cv_to_transform:
            sets_tf.append(tf_transformer.transform(dataset))
    else:
        sets_tf.append(tf_transformer.transform(sets_cv_to_transform))

    print("Terminé !")
    print('Nombre de sets retournés : ' + str(len(sets_tf)))
    return sets_tf


def tokenizing_and_tfidf(
        set_to_fit,
        sets_to_transform,
        max_feat=None,
        mdf=1.0):
    """
        Concatène les deux opérations et crée
        un BoW+TF-IDF.
    """

    cv = tokenizing(set_to_fit, sets_to_transform, max_feat, mdf)
    nbr_sets = len(cv)
    if (nbr_sets != 2):
        return __tfidf(cv[0], cv[1:nbr_sets])
    else:
        return __tfidf(cv[0], cv[1])


def my_MultinomialNB(X_train, X_test, y_train, y_test, a=1.0, cross_val=5):
    """
        Entraîne un classifier MultinomialNB.
    """
    clf = MultinomialNB(alpha=a)
    print("Entraînement...")
    clf.fit(X_train, y_train)
    print("Terminé !\n")
    print(
        "Evaluation par Cross-Validation (" +
        str(cross_val) +
        ") avec alpha = " +
        str(a) +
        " :")
    cross_validation_score = np.mean(
        cross_val_score(
            clf,
            X_train,
            y_train,
            cv=cross_val))
    print(cross_validation_score)
    print('')
    print("Précision/Score sur les données de Test : ")
    score = clf.score(X_test, y_test)
    print(score)
    y_pred = clf.predict(X_test)
    print("")
    print("Matrice de Confusion :")
    print(confusion_matrix(y_pred, y_test))
    print("")
    print("Rapport de classification :")
    print(classification_report(y_pred, y_test))
    return [clf, a, cross_validation_score, score]


def Grid_Search_CV_MultinomialNB(X_train, y_train, nb_crossval=3, tfidf=True):
    """
        Performe une GridSearchCV pour un classifier
        MultinomialNB.
    """
    if (tfidf):
        pipeline_gscv = Pipeline([
            ('vector', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

    else:
        pipeline_gscv = Pipeline([
            ('vector', CountVectorizer()),
            ('clf', MultinomialNB()),
        ])

    parametres = {
        'vector__max_df': (0.5, 0.7, 0.75, 0.8),
        'vector__max_features': (1000, 1500, 2000),
        'clf__alpha': (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0)
    }

    gs_cv = GridSearchCV(
        pipeline_gscv,
        parametres,
        verbose=1,
        n_jobs=-1,
        cv=nb_crossval)

    print("Grid Search MultinomialNB en cours...")
    print("Pipeline à suivre :", [name for name, _ in pipeline_gscv.steps])
    print("Paramètres à tester:")
    pprint(parametres)
    gs_cv.fit(X_train, y_train)

    print("Terminé !")
    print('')

    print("Meilleurs paramètres : ")
    best_parameters = gs_cv.best_estimator_.get_params()

    for param_name in sorted(parametres.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("\nMeilleur score: %0.3f" % gs_cv.best_score_)

    return best_parameters


def my_MLP(
        X_train,
        X_test,
        y_train,
        y_test,
        al=0.0001,
        activ='relu',
        hdsizes=100,
        verb=True,
        early_stop=True,
        bsize='auto'):
    """
        Entraîne un classifier MLP.
    """
    mlp = MLPClassifier(
        alpha=al,
        activation=activ,
        hidden_layer_sizes=hdsizes,
        verbose=verb,
        early_stopping=early_stop,
        batch_size=bsize,
        random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print('')
    print("Précision/Score sur les données de Test : ")
    score = mlp.score(X_test, y_test)
    print(score)
    print('')
    print(classification_report(y_pred, y_test))
    print('')
    print(confusion_matrix(y_pred, y_test))
    return [mlp, al, activ, hdsizes, early_stop, bsize, score]


def Grid_Search_CV_MLP(X_train, y_train, nb_crossval=3):
    """
        Performe une GridSearchCV pour un classifier
        MLP.
    """
    pipeline_gscv = Pipeline([
        ('mlp', MLPClassifier(random_state=42)),
    ])

    parametres = {
        'mlp__batch_size': (50, 100, 150),
        'mlp__activation': ('relu', 'tanh', 'logistic'),
    }

    gs_cv = GridSearchCV(
        pipeline_gscv,
        parametres,
        verbose=1,
        n_jobs=-1,
        cv=nb_crossval)

    print("Grid Search MLP en cours...")
    print("Pipeline à suivre :", [name for name, _ in pipeline_gscv.steps])
    print("Paramètres à tester:")
    pprint(parametres)
    gs_cv.fit(X_train, y_train)

    print("Terminé !")
    print('')

    print("Meilleurs paramètres : ")
    best_parameters = gs_cv.best_estimator_.get_params()

    for param_name in sorted(parametres.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("\nMeilleur score: %0.3f" % gs_cv.best_score_)

    return best_parameters


def main():
    # Chargement des données
    csv_path = 'data/Tobacco3482.csv'
    data = pd.read_csv(csv_path, sep=",")

    # Statistiques
    print('Quelques Statistiques :')
    data.describe(include='all')
    print('\nSamples :')
    data.sample(10)
    repartition_numbers = data['label'].value_counts()
    print('\nRépartition :')
    print(repartition_numbers)
    repartition_numbers.plot.bar(title='Répartition des documents par label')
    dictionnary_word_average_number = count_average_number_of_words_per_class(
        'data/Tobacco3482-OCR/')
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(dictionnary_word_average_number)), list(
        dictionnary_word_average_number.values()), align='center')
    plt.xticks(range(len(dictionnary_word_average_number)),
               list(dictionnary_word_average_number.keys()))
    plt.show()

    # Split Data et Fabrication des Sets
    print('\nSplit Data et Fabrication des Sets :')
    datasets = organize_and_split_data(csv_path)
    X_train = datasets[0]
    X_val = datasets[1]
    X_test = datasets[2]
    X_trainval = datasets[3]
    y_train = datasets[4]
    y_val = datasets[5]
    y_test = datasets[6]
    y_trainval = datasets[7]

    # MultinomialNB - TF-IDF - Optimisé
    print('\n MultinomialNB - TF-IDF - Optimisé :\n')
    datasets_tokenized_idf = tokenizing_and_tfidf(
        X_trainval, X_test, 1000, 0.8)
    X_trainval_tf = datasets_tokenized_idf[0]
    X_test_tf = datasets_tokenized_idf[1]
    final_MulNB = my_MultinomialNB(
        X_trainval_tf, X_test_tf, y_trainval, y_test, 0.2)
    print('')

    # MultinomialNB - BoW - Optimisé
    print('\n MultinomialNB - BoW - Optimisé :\n')
    datasets_tokenized_bow = tokenizing(X_trainval, X_test, 2000, 0.75)
    X_trainval_cv = datasets_tokenized_bow[0]
    X_test_cv = datasets_tokenized_bow[1]
    final_MulNB_bow = my_MultinomialNB(
        X_trainval_cv, X_test_cv, y_trainval, y_test, 0.05)
    print('')

    # MLP - TF-IDF - Optimisé
    print('\n MLP - TF-IDF - Optimisé :\n')
    datasets_tokenized_idf = tokenizing_and_tfidf(
        X_trainval, X_test, 2000, 0.80)
    X_trainval_tf = datasets_tokenized_idf[0]
    X_test_tf = datasets_tokenized_idf[1]
    final_mlp_tfidf = my_MLP(
        X_trainval_tf,
        X_test_tf,
        y_trainval,
        y_test,
        0.0001,
        'logistic',
        100,
        True,
        True,
        50)
    print('')

    # MLP - BoW - Optimisé
    print('\n MLP - BoW - Optimisé :\n')
    datasets_tokenized_bow = tokenizing(X_trainval, X_test, 2000, 0.75)
    X_trainval_cv = datasets_tokenized_bow[0]
    X_test_cv = datasets_tokenized_bow[1]
    final_mlp_bow = my_MLP(
        X_trainval_cv,
        X_test_cv,
        y_trainval,
        y_test,
        0.0001,
        'relu',
        100,
        True,
        True,
        200)


if __name__ == '__main__':
    main()
