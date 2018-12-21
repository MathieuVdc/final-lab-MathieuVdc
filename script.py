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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline




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
    X_trainval = X_train + X_val
    y_trainval = y_train + y_val
    print("Fabriqué 4 jeux de données depuis les données : X_train, X_val, X_test, X_trainval (X_train+X_val) ")
    print("taille de train : "+str(len(X_train)))
    print("taille de val : "+str(len(X_val)))
    print("taille de trainval : "+str(len(X_trainval)))
    print("taille de test : "+str(len(X_test)))
    return [X_train, X_val, X_test, X_trainval, y_train, y_val, y_test, y_trainval]


def organize_and_split_data(input_csv_path):
    
    listXy = __data_organizing(input_csv_path)
    datasets = __data_splitting(listXy[0], listXy[1])
    return datasets



def __tokenizing(set_to_fit , sets_to_transform, max_feat = None, mdf=1.0):
    
    print("Tokenization...")
    vectorizer = CountVectorizer(max_features=max_feat, max_df = mdf)
    vectorizer.fit(set_to_fit)
    sets_cv = []
    sets_cv.append(vectorizer.transform(set_to_fit))
    
    if (isinstance(sets_to_transform[0],list)):
        for dataset in sets_to_transform :
            sets_cv.append(vectorizer.transform(dataset))
    else :
        sets_cv.append(vectorizer.transform(sets_to_transform))
               
    print("Terminé !\n")
    print("Taille du vocabulaire :")
    print(str(len(vectorizer.get_feature_names())))
    print('')
    print('Nombre de sets retournés : '+str(len(sets_cv)))
    return sets_cv


def __tfidf(set_cv_to_fit, sets_cv_to_transform):
    
    print("TFIDF...")
    tf_transformer = TfidfTransformer().fit(set_cv_to_fit)
    sets_tf = []
    sets_tf.append(tf_transformer.transform(set_cv_to_fit))
    
    if (isinstance(sets_cv_to_transform[0],list)):
        for dataset in sets_cv_to_transform :
            sets_tf.append(tf_transformer.transform(dataset))
    else:
        sets_tf.append(tf_transformer.transform(sets_cv_to_transform))       
        
    print("Terminé !")
    print('Nombre de sets retournés : '+str(len(sets_tf)))
    return sets_tf


def tokenizing_and_tfidf(set_to_fit, sets_to_transform, max_feat = None, mdf = 1.0):
    
    cv = __tokenizing(set_to_fit, sets_to_transform, max_feat, mdf)
    nbr_sets = len(cv)
    if (nbr_sets != 2) :
        return __tfidf(cv[0], cv[1:nbr_sets])
    else :
        return __tfidf(cv[0], cv[1])
 
    
def my_MultinomialNB(X_train, X_test, y_train, y_test, a = 1.0, cross_val = 5):
    
    clf = MultinomialNB(alpha=a)
    print("Entraînement...")
    clf.fit(X_train,y_train)
    print("Terminé !\n")
    print("Evaluation par Cross-Validation ("+str(cross_val)+") avec alpha = "+str(a)+" :")
    cross_validation_score = np.mean(cross_val_score(clf, X_train, y_train, cv=cross_val))
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


def Grid_Search_CV_MultinomialNB(X_train, y_train, nb_crossval=3):    

    #logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    pipeline_gscv = Pipeline([
    ('vector', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
    ])

    parametres = {
    'vector__max_df': (0.1, 0.2, 0.5, 0.7, 0.75, 0.8),
    'vector__max_features' : (500, 1000, 1500, 2000),
    'clf__alpha': (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0)
    }
    
    gs_cv = GridSearchCV(pipeline_gscv, parametres, verbose=1, n_jobs=-1,  cv=nb_crossval)

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

            
    