#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:10:46 2021

@author: Anıl Alper
"""
from feature_selection import X_all, y_all, anova_selected_all, pca_selected_all, X_ethos, y_ethos, anova_selected_ethos, pca_selected_ethos, X_filtered, y_filtered, anova_selected_filtered, pca_selected_filtered, cocogen_full, binary_dataset, ethos_data, ethos_labels, filtered_X, filtered_y
from Classical_ML_Methods import pick_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def pick_features(feature_selection = None, data_type = "all"):
    if data_type == "ethos":
        if feature_selection == "ANOVA":   
            features = anova_selected_ethos
        elif feature_selection == "PCA":
            features = pca_selected_ethos
        else:
            features = ethos_data.columns.tolist()
    elif data_type == "off":
        if feature_selection == "ANOVA":   
            features = anova_selected_filtered
        elif feature_selection == "PCA":
            features = pca_selected_filtered
        else:
            features = filtered_X.columns.tolist()
    else: 
        if feature_selection == "ANOVA":   
            features = anova_selected_all
        elif feature_selection == "PCA":
            features = pca_selected_all
        else:
            features = cocogen_full.columns.tolist()
            
    return features

def get_model(feature_selection = None, data_type="all"):
    
    features = pick_features(feature_selection, data_type)
    
    model = tf.keras.models.Sequential([
                
        #Hidden Layer
        tf.keras.layers.Dense(100, activation = "relu", input_shape=(len(features),), ),
        
        tf.keras.layers.Dropout(0.2),

        #Hidden Layer
        tf.keras.layers.Dense(50, activation = "relu"),
        
        tf.keras.layers.Dropout(0.2),

        #Output Layer
        tf.keras.layers.Dense(1, activation = "sigmoid")
        
        ])
    
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = ["binary_accuracy", tf.keras.metrics.FalseNegatives(thresholds=0.5), tf.keras.metrics.FalsePositives(thresholds=0.5), tf.keras.metrics.TrueNegatives(thresholds=0.5), tf.keras.metrics.TruePositives(thresholds=0.5)]
        )
    
    return model

def return_model_results(data_X, data_Y, feature_selection = None, data_type="all"):
   
    data_X = pick_data(feature_selection, data_X, data_type)
    model = get_model(feature_selection, data_type)
    
    X = data_X.to_numpy()
    Y = data_Y.to_numpy()
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42, stratify = Y
    )
    
    model.fit(x_train, y_train, epochs = 50)
    results = model.evaluate(x_test, y_test)
    
    confusion_matrix = dict()
    confusion_matrix["fn"] = results[2]
    confusion_matrix["fp"] = results[3]
    confusion_matrix["tn"] = results[4]
    confusion_matrix["tp"] = results[5]
    
    f1_score = confusion_matrix["tp"] / (confusion_matrix["tp"] + 0.5 * (confusion_matrix["fp"] + confusion_matrix["fn"]))
    
    return (f1_score, confusion_matrix, feature_selection)

def return_oversampled_model(data_X, data_Y, percentage, feature_selection = None, data_type="all"):
    
    data_X = pick_data(feature_selection, data_X, data_type)
    model = get_model(feature_selection, data_type)
    
    X = data_X.to_numpy()
    y = data_Y.to_numpy()
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    x_training, x_test, y_training, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=data_Y
        )
    
    x_train, x_val, y_train, y_val = train_test_split(x_training, y_training, 
                                                      test_size = 0.2, 
                                                      random_state = 42)
    
    sm = SMOTE(random_state = 12, sampling_strategy = percentage)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    
    model.fit(x_train_res, y_train_res, epochs = 50)
    results = model.evaluate(x_test, y_test)
    
    confusion_matrix = dict()
    confusion_matrix["fn"] = results[2]
    confusion_matrix["fp"] = results[3]
    confusion_matrix["tn"] = results[4]
    confusion_matrix["tp"] = results[5]
    
    f1_score = confusion_matrix["tp"] / (confusion_matrix["tp"] + 0.5 * (confusion_matrix["fp"] + confusion_matrix["fn"]))
    
    return (f1_score, confusion_matrix, feature_selection)
    
'''
#Trials (HS_Offensive + Ethos [offensive texts included])
print("HS_Offensive + Ethos (Offensive texts included)")
print("-------------------------------------------------------------")
print("Neural Networks Performance:")
results_all = list()   
results_all.append(return_model_results(X_all, y_all, "ANOVA"))
results_all.append(return_oversampled_model(X_all, y_all, 0.4, "ANOVA"))
results_all.append(return_model_results(X_all, y_all,"PCA"))
results_all.append(return_oversampled_model(X_all, y_all, 0.4, "PCA"))
results_all.append(return_model_results(cocogen_full, binary_dataset))
results_all.append(return_oversampled_model(cocogen_full, binary_dataset, 0.4))
print(results_all)
print("-------------------------------------------------------------")


#Trials With Only Ethos Dataset
print("Trials With Only Ethos Dataset")
print("-------------------------------------------------------------")
print("Neural Networks Performance:")
results_ethos = list()
results_ethos.append(return_model_results(X_ethos, y_ethos, "ANOVA", "ethos"))
results_ethos.append(return_model_results(X_ethos, y_ethos, "PCA", "ethos"))
results_ethos.append(return_model_results(ethos_data, ethos_labels, data_type="ethos"))
print(results_ethos)



#Trials With Offensive and Ethos (Offensive Texts Removed)
print("HS_Offensive + Ethos (Offensive Texts Removed)")
print("-------------------------------------------------------------")
print("Neural Networks Performance:")
results_filtered = list()
results_filtered.append(return_model_results(X_filtered, y_filtered, "ANOVA", "off"))
results_filtered.append(return_model_results(X_filtered, y_filtered, "PCA", "off"))
results_filtered.append(return_model_results(filtered_X, filtered_y, data_type="off"))
print(results_filtered)
'''
