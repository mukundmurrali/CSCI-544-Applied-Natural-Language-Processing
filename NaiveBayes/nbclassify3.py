# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 20:16:34 2019

@author: mukun
"""

import normalize
import json
import os
import nbtesthelper
import glob
import sys

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
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

file = open('nbmodel.txt', 'r')

classVsWordProbablities = json.loads(file.read())
result={}
file = open("nboutput.txt","w") 
for i in range(len(paths)):
    content = open(paths[i]).read()
    content = normalize.normalizeData(content)
    res = nbtesthelper.find_class(classVsWordProbablities, content, paths[i])
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
#print(result)

