#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 20:05:34 2021

@author: Anıl Alper
"""
# followed the instructions on https://machinelearningmastery.com/calculate-feature-importance-with-python/
# https://medium.com/cascade-bio-blog/creating-visualizations-to-better-understand-your-data-and-models-part-1-a51e7e5af9c0
# https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155
# ANOVA analysis for feature selection

import pandas as pd 
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ANOVA
def Important_Features_ANOVA(X_fs, y_fs):
    
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_fs)
    X_fs = X_fs[X_fs.columns[constant_filter.get_support()]]
    
    
    fs = SelectKBest(score_func = f_classif, k = "all")
    fs.fit(X_fs, y_fs)
        
    anova_selected_features = list()
    selected_z_scores = list()
    
    '''
    #the graph of all features
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.title("Z-Scores of Cocogen Features")
    plt.xlabel("Features")
    plt.ylabel("Z-Scores")
    plt.show()
    '''
    
    for i in range(len(fs.pvalues_)):
        if fs.pvalues_[i] < 0.05:
            anova_selected_features.append(X_fs.columns[i])
            selected_z_scores.append(fs.scores_[i])
    
    
    '''
    anova_selected_frame = pd.DataFrame(data = {"Attributes": [anova_selected_features[i] for i in range(len(selected_z_scores))],
                                            "F-Scores": selected_z_scores})
    anova_selected_frame = anova_selected_frame.sort_values(by = "F-Scores", ascending = False)
    plt.bar(x = anova_selected_frame["Attributes"], height = anova_selected_frame["F-Scores"])
    plt.title("Z-Scores of The Selected Cocogen Features")
    plt.xlabel("Features")
    plt.ylabel("Z-Scores")
    plt.xticks(rotation = "vertical", fontsize = 12)
    plt.show()
    '''
            
    return (anova_selected_features, selected_z_scores)


#edited the code from https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155
#----------------------------------------

# PCA
def Important_Features_PCA(X_fs, Y_fs, top):
    
    ss = StandardScaler()

    X_feature_scaled = ss.fit_transform(X_fs)
    
    pca = PCA().fit(X_feature_scaled)
    
    '''
    plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
    plt.title('Cumulative explained variance by number of principal components', size=20)
    plt.show()  
    '''

    if (X_fs.shape[0] >= X_fs.shape[1]):
        loadings = pd.DataFrame (
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i}' for i in range(1, len(X_fs.columns) + 1)],
        index=X_fs.columns
        )
    else:
        loadings = pd.DataFrame (
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i}' for i in range(1, X_fs.shape[0] + 1)],
        index=X_fs.columns
        )

    pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
    pc1_loadings = pc1_loadings.reset_index()
    pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']
    
    ngram_counter = dict()
    pca_selected_features = list()
    for index, row in pc1_loadings.iterrows():
       feature = row["Attribute"]
       if "ngram.Simplified" in feature:
           category = feature.split("_reg_")[1].split("_threshold")[0]
           ngram_type = re.search("ngram_[0-9]", feature)[0]
           if (ngram_type, category) not in ngram_counter:
               ngram_counter[(ngram_type, category)] = 1
               pca_selected_features.append(feature)
           else:
               if ngram_counter[(ngram_type, category)] == 1:
                   pca_selected_features.append(feature)
                   ngram_counter[(ngram_type, category)] += 1
       else:
           pca_selected_features.append(feature)
    
    return pca_selected_features[:top]

def print_distribution(selected_features, feat):
    features_dict = dict()
    for ft in selected_features:
        feature = ft.split(".")[0]
        if feature not in features_dict:
            features_dict[feature] = 1
        else:
            features_dict[feature] += 1
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    features = list(features_dict.keys())
    numbers = list(features_dict.values())
    total = sum(numbers)
    
    ax.pie(numbers, labels = features, autopct=lambda p: '{:.0f}'.format(p * total / 100), textprops={'fontsize': 18})

    plt.title("Distribution of Features (%s)" % (feat,), fontsize=18)
    
    plt.show()
    
    return features_dict

# Ethos + HS_Offensive (offensive texts included)
binary_dataset = pd.read_csv("Dataset/HateSpeech_Binary_Dataset.csv", usecols=["isHate"])
cocogen_full = pd.read_csv("cocogen_full_data.csv")
cocogen_full.drop(["Unnamed: 0", "comment"], inplace = True, axis = 1)


X_feature, X_all, y_feature, y_all = train_test_split(cocogen_full, binary_dataset["isHate"], test_size = 0.8, random_state = 2, stratify=binary_dataset["isHate"])
anova_selected_all = Important_Features_ANOVA(X_feature, y_feature)[0]
pca_selected_all = Important_Features_PCA(X_feature, y_feature, 100)

#---------------------------------------------------------------------------

# Ethos (offensive texts included)
hs_binary = pd.read_csv("Dataset/HateSpeech_Binary_Dataset.csv")
ethos = hs_binary.loc[hs_binary["dataset"] == "ethos"]

ethos_indices = ethos["Unnamed: 0"].tolist()
ethos_data = cocogen_full.loc[ethos_indices]
ethos_labels = hs_binary["isHate"].loc[ethos_indices]


X_ft_ethos, X_ethos, y_ft_ethos, y_ethos = train_test_split(ethos_data, ethos_labels, test_size = 0.8, random_state = 2, stratify=ethos_labels)
anova_selected_ethos = Important_Features_ANOVA(X_ft_ethos, y_ft_ethos)[0]
pca_selected_ethos = Important_Features_PCA(X_ft_ethos, y_ft_ethos, 100)
#---------------------------------------------------------------------------

# Ethos + HS_Offensive (offensive texts not included)
hs_offensive = pd.read_csv("Dataset/HS_Offensive.csv", usecols = ['class', 'tweet'])
hs_offensive = hs_offensive.loc[hs_offensive["class"] != 1] 
hs_offensive_indices = [x+len(ethos) for x in list(hs_offensive.index)]

index_list = ethos_indices + hs_offensive_indices
filtered_X = cocogen_full.loc[index_list]
filtered_y = hs_binary["isHate"].loc[index_list]

X_ft_filtered, X_filtered, y_ft_filtered, y_filtered = train_test_split(filtered_X, filtered_y, test_size = 0.8, random_state = 2, stratify=filtered_y)
anova_selected_filtered = Important_Features_ANOVA(X_ft_filtered, y_ft_filtered)[0]
pca_selected_filtered = Important_Features_PCA(X_ft_filtered, y_ft_filtered, 100)
