#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:13:11 2021

@author: Anıl Alper
"""

# Source: https://kiwidamien.github.io/how-to-do-cross-validation-when-upsampling-data.html
#         https://www.askpython.com/python/examples/k-fold-cross-validation

from feature_selection import X_all, y_all, anova_selected_all, pca_selected_all, X_ethos, y_ethos, anova_selected_ethos, pca_selected_ethos, X_filtered, y_filtered, anova_selected_filtered, pca_selected_filtered, cocogen_full, binary_dataset, ethos_data, ethos_labels, filtered_X, filtered_y
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMax
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd

k = 10
skf = StratifiedKFold(n_splits = k)
ss = StandardScaler()

def pick_data(feature_selection, data_X, data_type):
    
    if data_type == "ethos":
        if feature_selection == "ANOVA":   
            selected_X = data_X[anova_selected_ethos]
        elif feature_selection == "PCA":
            selected_X = data_X[pca_selected_ethos]
        else:
            selected_X = data_X
    elif data_type == "off":
        if feature_selection == "ANOVA":   
            selected_X = data_X[anova_selected_filtered]
        elif feature_selection == "PCA":
            selected_X = data_X[pca_selected_filtered]
        else:
            selected_X = data_X
    else: 
        if feature_selection == "ANOVA":   
            selected_X = data_X[anova_selected_all]
        elif feature_selection == "PCA":
            selected_X = data_X[pca_selected_all]
        else:
            selected_X = data_X
            
    return selected_X

def test_model_feature(model, data_X, data_Y, feature_selection=None, data_type="all"):
   
   selected_X = pick_data(feature_selection, data_X, data_type) 
    
   X_scaled = ss.fit_transform(selected_X)
    
   model_score = cross_val_score(model, X_scaled, data_Y.to_numpy().ravel(), cv=skf, scoring="f1")

   return model_score.mean()

def test_model_feature_cv(model, data_X, data_Y, feature_selection = None, data_type="all"):
    
    X = pick_data(feature_selection, data_X, data_type)
    y = data_Y
    f1_scores = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
        
        print(y_train)
        model.fit(X_train,y_train.to_numpy().ravel())
        print(y_train.to_numpy().ravel())
        pred_values = model.predict(X_test)
        
        acc = f1_score(pred_values, y_test)
        f1_scores.append(acc)
    
    avg_acc_score = sum(f1_scores)/k
    print('accuracy of each fold - {}'.format(f1_scores))
    print('Avg accuracy : {}'.format(avg_acc_score))
    
def test_oversampled_model(model, data_X, data_Y, feature_selection=None, data_type="all"):
   
    selected_X = pick_data(feature_selection, data_X, data_type)
    steps = [("oversample", RandomOverSampler(sampling_strategy=0.25)), ("model", model)]
    pipeline = Pipeline(steps=steps)
    
    X_scaled = ss.fit_transform(selected_X)
    
    model_score = cross_val_score(pipeline, X_scaled, data_Y.to_numpy().ravel(), cv=skf, scoring="f1")
    return model_score.mean()

def model_test(model, data_X, data_Y, feature_selection=None, data_type="all"):
    
    data_X = pick_data(feature_selection, data_X, data_type)
    data_Y = data_Y.to_numpy().ravel()
    
    x_training, x_test, y_training, y_test = train_test_split(
        data_X, data_Y, test_size=0.2, random_state = 42, stratify=data_Y
    )
  
    y_pred = model.fit(x_training, y_training).predict(x_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_matrix_dict = dict()
    confusion_matrix_dict["tn"] = tn
    confusion_matrix_dict["fp"] = fp
    confusion_matrix_dict["fn"] = fn
    confusion_matrix_dict["tp"] = tp
    
    return (f1_score(y_test, y_pred), confusion_matrix_dict)

def model_test_oversampled(model, data_X, data_Y, percentage, feature_selection = None, data_type="all"):
    
    data_X = pick_data(feature_selection, data_X, data_type)
    data_Y = data_Y.to_numpy().ravel()
    
    x_train, x_test, y_train, y_test = train_test_split(
        data_X, data_Y, test_size=0.2, random_state=42, stratify=data_Y
        )
    
    
    sm = SMOTE(random_state = 12, sampling_strategy = percentage)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    
    y_pred = model.fit(x_train_res, y_train_res).predict(x_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_matrix_dict = dict()
    confusion_matrix_dict["tn"] = tn
    confusion_matrix_dict["fp"] = fp
    confusion_matrix_dict["fn"] = fn
    confusion_matrix_dict["tp"] = tp
    
    return (f1_score(y_test, y_pred), confusion_matrix_dict)


'''
#Trials (HS_Offensive + Ethos [offensive texts included])

print("HS_Offensive + Ethos (Offensive texts included)")
print("-------------------------------------------------------------")
# Naive Bayes
print("Naive Bayes Performance:")
print("ANOVA:")
print(str(model_test(GaussianNB(), X_all, y_all, "ANOVA")))
#print(str(model_test_oversampled(GaussianNB(), X_all, y_all, 0.2, "ANOVA")))

print("PCA")
print(str(model_test(GaussianNB(), X_all, y_all, "PCA")))
#print(str(model_test_oversampled(GaussianNB(), X_all, y_all, 0.2, "PCA")))

print("No Feature")
print(str(model_test(GaussianNB(), cocogen_full, binary_dataset)))
#print(str(model_test_oversampled(GaussianNB(), cocogen_full, binary_dataset, 0.2)))
print("------------")


# Logistic Regression
print("Logistic Regression Performance:") 
print("ANOVA:")
print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_all, y_all, "ANOVA")))
#print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_all, y_all, 0.2, "ANOVA")))

print("PCA:")
print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_all, y_all, "PCA")))
#print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_all, y_all, 0.2, "PCA")))

print("No Feature")
print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), cocogen_full, binary_dataset)))
#print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), cocogen_full, binary_dataset, 0.2)))
print("------------")

# Support Vector Machine (SVM)
print("Support Vector Machine Performance:")
print("ANOVA:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), X_all, y_all, "ANOVA")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), X_all, y_all, 0.2, "ANOVA")))

print("PCA:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), X_all, y_all, "PCA")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), X_all, y_all, 0.2, "PCA")))

print("------------")
print("No Feature:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), cocogen_full, binary_dataset)))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), cocogen_full, binary_dataset, 0.2)))
print("------------")



# Random Forest Classifier
print("Random Forest Performance:")
print("ANOVA:")
#print(str(model_test(RandomForestClassifier(), X_all, y_all, "ANOVA")))
print(str(model_test_oversampled(RandomForestClassifier(), X_all, y_all, 0.2, "ANOVA")))

print("PCA:")
#print(str(model_test(RandomForestClassifier(), X_all, y_all, "PCA")))
print(str(model_test_oversampled(RandomForestClassifier(), X_all, y_all, 0.2, "PCA")))

print("No Feature Selection")
#print(str(model_test(RandomForestClassifier(), cocogen_full, binary_dataset)))
print(str(model_test_oversampled(RandomForestClassifier(), cocogen_full, binary_dataset, 0.2)))
print("----------------------------------------------------------------------")


#Trials With Only Ethos Dataset

print("Trials With Only Ethos Dataset")
print("-------------------------------------------------------------")
# Naive Bayes
print("Naive Bayes Performance:")
print("ANOVA:")
#print(str(model_test(GaussianNB(), X_ethos, y_ethos, "ANOVA", "ethos")))
print(str(model_test_oversampled(GaussianNB(), X_ethos, y_ethos, 0.2, "ANOVA", "ethos")))

print("PCA")
#print(str(model_test(GaussianNB(), X_ethos, y_ethos, "PCA", "ethos")))
print(str(model_test_oversampled(GaussianNB(), X_ethos, y_ethos, 0.2, "PCA", "ethos")))

print("No Feature")
#print(str(model_test(GaussianNB(), ethos_data, ethos_labels, data_type="ethos")))
print(str(model_test_oversampled(GaussianNB(), ethos_data, ethos_labels, 0.2, data_type="ethos")))
print("------------")


# Logistic Regression
print("Logistic Regression Performance:") 
print("ANOVA:")
#print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_ethos, y_ethos, "ANOVA", "ethos")))
print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_ethos, y_ethos, 0.2, "ANOVA", "ethos")))

print("PCA:")
#print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_ethos, y_ethos, "PCA", "ethos")))
print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_ethos, y_ethos, 0.2, "PCA", "ethos")))

print("No Feature")
#print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), ethos_data, ethos_labels, data_type="ethos")))
print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), ethos_data, ethos_labels, 0.2, data_type="ethos")))
print("------------")


# Support Vector Machine (SVM)
print("Support Vector Machine Performance:")
print("ANOVA:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), X_ethos, y_ethos, "ANOVA", "ethos")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), X_ethos, y_ethos, 0.2, "ANOVA", "ethos")))

print("PCA:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), X_ethos, y_ethos, "PCA", "ethos")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), X_ethos, y_ethos, 0.2, "PCA"), "ethos"))

print("------------")
print("No Feature:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), ethos_data, ethos_labels, data_type="ethos")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), ethos_data, ethos_labels, 0.2, data_type="ethos")))
print("------------")


# Random Forest Classifier
print("Random Forest Performance:")
print("ANOVA:")
#print(str(model_test(RandomForestClassifier(), X_ethos, y_ethos, "ANOVA", "ethos")))
print(str(model_test_oversampled(RandomForestClassifier(), X_ethos, y_ethos, 0.2, "ANOVA", "ethos")))

print("PCA:")
#print(str(model_test(RandomForestClassifier(), X_ethos, y_ethos, "PCA", "ethos")))
print(str(model_test_oversampled(RandomForestClassifier(), X_ethos, y_ethos, 0.2, "PCA", "ethos")))

print("No Feature Selection")
#print(str(model_test(RandomForestClassifier(), ethos_data, ethos_labels, data_type="ethos")))
print(str(model_test_oversampled(RandomForestClassifier(), ethos_data, ethos_labels, 0.2, data_type="ethos")))
print("----------------------------------------------------------------------")



#Trials With Offensive and Ethos (Offensive Texts Removed)

print("HS_Offensive + Ethos (Offensive Texts Removed)")
print("-------------------------------------------------------------")

# Naive Bayes
print("Naive Bayes Performance:")
print("ANOVA:")
#print(str(model_test(GaussianNB(), X_filtered, y_filtered, "ANOVA", "off")))
print(str(model_test_oversampled(GaussianNB(), X_filtered, y_filtered, 0.2, "ANOVA", "off")))

print("PCA")
#print(str(model_test(GaussianNB(), X_filtered, y_filtered, "PCA", "off")))
print(str(model_test_oversampled(GaussianNB(), X_filtered, y_filtered, 0.2, "PCA", "off")))

print("No Feature")
#print(str(model_test(GaussianNB(), filtered_X, filtered_y, data_type="off")))
print(str(model_test_oversampled(GaussianNB(),filtered_X, filtered_y, 0.2, data_type="off")))
print("------------")


# Logistic Regression
print("Logistic Regression Performance:") 
print("ANOVA:")
#print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_filtered, y_filtered, "ANOVA", "off")))
print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_filtered, y_filtered, 0.2, "ANOVA", "off")))

print("PCA:")
#print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_filtered, y_filtered, "PCA", "off")))
print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), X_filtered, y_filtered, 0.2, "PCA", "off")))

print("No Feature")
#print(str(model_test(LogisticRegression(max_iter = 30000, class_weight = "balanced"), filtered_X, filtered_y, data_type = "off")))
print(str(model_test_oversampled(LogisticRegression(max_iter = 30000, class_weight = "balanced"), filtered_X, filtered_y, 0.2, data_type = "off")))
print("------------")


# Support Vector Machine (SVM)
print("Support Vector Machine Performance:")
print("ANOVA:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), X_filtered, y_filtered, "ANOVA", "off")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), X_filtered, y_filtered, 0.2, "ANOVA", "off")))

print("PCA:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), X_filtered, y_filtered, "PCA", "off")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), X_filtered, y_filtered, 0.2, "PCA", "off")))

print("------------")
print("No Feature:")
#print(str(model_test(svm.SVC(kernel="sigmoid"), filtered_X, filtered_y, data_type="off")))
print(str(model_test_oversampled(svm.SVC(kernel="sigmoid"), filtered_X, filtered_y, 0.2, data_type="off")))
print("------------")


# Random Forest Classifier
print("Random Forest Performance:")
print("ANOVA:")
#print(str(model_test(RandomForestClassifier(), X_filtered, y_filtered, "ANOVA", "off")))
print(str(model_test_oversampled(RandomForestClassifier(), X_filtered, y_filtered, 0.2, "ANOVA", "off")))

print("PCA:")
#print(str(model_test(RandomForestClassifier(), X_filtered, y_filtered, "PCA", "off")))
print(str(model_test_oversampled(RandomForestClassifier(), X_filtered, y_filtered, 0.2, "PCA", "off")))

print("No Feature Selection")
#print(str(model_test(RandomForestClassifier(), filtered_X, filtered_y, data_type="off")))
print(str(model_test_oversampled(RandomForestClassifier(), filtered_X, filtered_y, 0.2, data_type="off")))
print("----------------------------------------------------------------------")
'''