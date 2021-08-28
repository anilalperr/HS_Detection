#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 17:52:41 2021

@author: Anıl Alper
"""

import os
import io
from sklearn.feature_extraction.text import CountVectorizer
from feature_selection import X_all, y_all, X_ft_ethos, y_ft_ethos, X_ft_filtered, y_ft_filtered

#tf-idf
def extract_words(directory):
    texts_list = list()
    for file in os.listdir(directory):
        file_directory = directory + os.sep + file
        with io.open(file_directory, "r", encoding="windows-1252") as text_file:
            data = text_file.read()
            texts_list.append(data)
    return texts_list
         

