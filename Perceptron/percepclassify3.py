# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 20:16:34 2019

@author: mukun
"""

import normalize
import json
import os
import glob
import sys
import perceptronhelper

all_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))
paths=[]
class_name=[]
for f in all_files:
      class1, class2, fold, fname = f.split('/')[-4:]
      if class1 in "positive_polarity":
          if class2 in "truthful_from_TripAdvisor":
              class_name.append(1)
          else:
              class_name.append(2)
      else:
          if class2 in "truthful_from_Web":
              class_name.append(3)
          else:
              class_name.append(4)
      paths.append(f)

fileObj = open(sys.argv[1],'r')
fileData = json.loads(fileObj.read())

weights_true_deceptive = fileData["weights_True_Deceptive"]
bias_True_Deceptive = fileData["bias_True_Deceptive"]
weights_Positive_Negative = fileData["weights_Positive_Negative"]
bias_Positive_Negative = fileData["bias_Positive_Negative"]
result={}
                
file = open("percepoutput.txt","w") 
for i in range(len(paths)):
    content = open(paths[i]).read()
    content = normalize.normalizeData(content)
    res = perceptronhelper.find_class(weights_true_deceptive,bias_True_Deceptive,
                                  weights_Positive_Negative, bias_Positive_Negative, 
                                  content, paths[i])
    if res == "1":
        file.write("truthful positive " + paths[i])
    if res == "2":
        file.write("deceptive positive " + paths[i])
    if res == "3":
        file.write("truthful negative " + paths[i])
    if res == "4":
        file.write("deceptive negative " + paths[i])
    file.write("\n")
    resMap = result.get(class_name[i], {})
    currTotal = resMap.get(res, 0)
    resMap[res] =  currTotal + 1;
    result[class_name[i]] = resMap
file.close()
c1 = result[1]["1"]
c2 = result[2]["2"]
c3 = result[3]["3"]
c4 = result[4]["4"]
ans = c1 + c2 + c3 + c4
