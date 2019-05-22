# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:12:13 2019

@author: mukun
"""
import math
def find_class(classVsWordProbablities, content, filename):
    words = content.split()
    class_results={}
    for key,value in classVsWordProbablities.items():
        probablity = 0
        for word in words:
            val = classVsWordProbablities.get(key).get(word, None)
            if val != None:
                val = math.log(val)
                probablity += val
        class_results[key] = probablity * 1/4
    #print(class_results)
    max = float('-inf')
    class_res = 0
    for key,value in class_results.items():
        if value > max :
            max = value
            class_res = key
#    print(filename + str(class_res))
    return class_res
