#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:21:09 2021

@author: student
"""
import pandas as pd 
import os
import re

def print_components(filename):
    file_measure = pd.read_csv("output_dir" + os.sep + filename)
    for name, values in file_measure.iteritems():
        if not "ngram." in name:
            print("{name}: {value}".format(name=name, value=values[0]))
    pass


frame_dict = dict()
frame_dict["comment"] = list()
total_text_num = 25781
for t in range(1,total_text_num+1):
    print(t)
    frame_dict["comment"].append("text{}".format(t))
    csv_file = pd.read_csv("output_dir" + os.sep + "text{}.csv".format(t))
    for name, values in csv_file.iteritems():
        #if "ngram." in name:
         #   break
        if name not in frame_dict:
            frame_dict[name] = list()
        frame_dict[name].append(values[0])

cocogen_data = pd.DataFrame(frame_dict)
cocogen_data.to_csv('cocogen_full_data.csv')




    