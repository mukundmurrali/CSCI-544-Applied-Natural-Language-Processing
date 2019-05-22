# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:12:13 2019

@author: mukun
"""
def find_class(weights_true_deceptive,bias_True_Deceptive,
                                  weights_Positive_Negative, bias_Positive_Negative, 
                                  content, path):
    words = content.split()
    wordCount={}
    for word in words:
        wordCount[word] = wordCount.get(word, 0) + 1;
    true_deceptive = 0;
    positive_negative = 0;
    for feature in wordCount:
        if feature in weights_true_deceptive:
            true_deceptive += weights_true_deceptive[feature] * wordCount[feature]
        if feature in weights_Positive_Negative:
            positive_negative += weights_Positive_Negative[feature] * wordCount[feature]
    true_deceptive += bias_True_Deceptive
    positive_negative += bias_Positive_Negative
    
    if positive_negative >= 0:
        if true_deceptive >= 0:
            return "1"
        else:
            return "2"

    else:
        if true_deceptive >= 0:
            return "3"
        else:
            return "4"
