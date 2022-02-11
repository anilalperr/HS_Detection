#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:45:49 2021
Last Edited Feb 10, 2022

@author: Anıl Alper
"""
import pandas as pd
import os
import re

# import the txt files as a data frame
hs_binary = pd.read_csv("Dataset/HateSpeech_Binary_Dataset.csv")

# drop the unnamed columns
hs_binary = hs_binary.loc[:, ~hs_binary.columns.str.contains('^Unnamed')]

#edited the code from https://www.sparkitecture.io/natural-language-processing/data-preparation
def preprocess(text):
    text = text.lower() # Make sure that all the letters are lowercase
    text = re.sub("(https?\://)\S+", "", text) # Remove links
    text = re.sub("(\\n)|\n|\r|\t", "", text) # Remove CR, tab, and LR
    text = re.sub("!!+|\"", "", text) # Remove extra exclamation points
    text = re.sub("\s+rt\s+", "", text) # Remove RTs
    text = re.sub("[^'.,!?+A-Za-z0-9_\s]([A-Za-z0-9_s.,!?])+", "", text) # Remove unnecessary punctuations
    text = re.sub("[^.,!?+A-Za-z0-9_\s]", "", text)  #Remove unnecessary punctuations
    text = re.sub("\.\.+", "", text) #Remove ...
    text = re.sub("\s\s+", " ", text) #Make sure that there is only one space character between each word
    text = re.sub(r"\s([.,!?])", r"\1", text) # Remove the space before any punctuation
    return text.strip()

#generates txt files for cocogen client
text_num = 0
texts_list = list()

#reads the binary dataset, preprocesses each text and add them to a new directory named input_dir
for index, row in hs_binary.iterrows():
    with open("input_dir/text{}.txt".format(text_num), "w+") as txt_file:
        new_text = preprocess(row["text"])
        if new_text[:3] == "rt ":
            new_text = new_text[3:]
        txt_file.write(new_text)
    texts_list.append(new_text)
    text_num += 1

# replace the previous texts with the preprocessed texts
hs_binary["text"] = texts_list

# create a path
path = r'Dataset'

# create a combined hate speech dataset
hs_binary.to_csv(os.path.join(path, r'combined_data.csv'), index=False)