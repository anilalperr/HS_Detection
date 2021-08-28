#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:45:49 2021

@author: Anıl Alper
"""
import pandas as pd
import re

hs_binary = pd.read_csv("Dataset/HateSpeech_Binary_Dataset.csv", usecols=["text"])


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
for index, row in hs_binary.iterrows():
    with open("input_dir/text{}.txt".format(text_num), "w+") as txt_file:
        new_text = preprocess(row["text"])
        if new_text[:3] == "rt ":
            new_text = new_text[3:]
        txt_file.write(new_text)
    texts_list.append(new_text)
    print(text_num)
    text_num += 1

