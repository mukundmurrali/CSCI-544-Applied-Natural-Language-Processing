# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:35:33 2019

@author: mukund
"""
import string
from string import digits

def normalizeData(data):
    data = lowerCase(data)
    data = removePunctuations(data)
    #data = removeDigits(data)
    return data

def lowerCase(data):
    result = data.lower()
    return result

def removeDigits(data):
     return ''.join(data for num in data if not num.isdigit())
    
def removePunctuations(data):
    return data.translate(str.maketrans('','',string.punctuation))
