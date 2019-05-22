#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import sys
import operator
# import data from file..

fileName = 'hmmmodel.txt'
fileObj = open(fileName, encoding='UTF-8')
fileData = json.loads(fileObj.read())
tagVsCount = fileData[0]
tagToTagCount = fileData[1]
wordVsTagCount = fileData[2]

#print(len(tagVsCount));

#tagVsCount = dict(sorted(tagVsCount.items(), key=operator.itemgetter(1), reverse=True)[:20])

# load data from file

path = sys.argv[1]
f = open(path, encoding='UTF-8')

# read from the file..

lines = f.readlines()

# for each sentence do this
finalResult=[]
for sentence in lines:

    # get the words..
    words = sentence.split()
    storeResult = {}  # 3d - matrix to store the data..
    for i in range(0, len(words) + 1):
        storeResult[i] = {}
        tagsToCheck = []
        if i == 0:
            word = words[i]
            if word in wordVsTagCount.keys():
                tagList = wordVsTagCount[word].keys()
                for tag in tagList:
                    storeResult[i][tag] = {}
                    storeResult[i][tag]['probablity'] = wordVsTagCount[word][tag] + tagToTagCount['start'][tag]
                    storeResult[i][tag]['backpointer'] = 'start'
            else:
                tagList = tagVsCount.keys()
                for tag in tagList:
                    if tag != 'start' and tag != 'end':
                        storeResult[i][tag] = {}
                        storeResult[i][tag]['probablity'] = tagToTagCount['start'][tag] ;
                        storeResult[i][tag]['backpointer'] = 'start'
            continue
        if i == len(words):
            maxValue = -sys.maxsize - 1
            result = ''
            for previousTag in storeResult[i - 1].keys():
                probablity = storeResult[i - 1][previousTag]['probablity'] +  tagToTagCount[previousTag]['end']
                if probablity > maxValue:
                    maxValue = probablity
                    result = previousTag
            storeResult[i]['end'] = {}
            storeResult[i]['end']['probablity'] = maxValue
            storeResult[i]['end']['backpointer'] = result
            continue
        word = words[i]
        if word in wordVsTagCount.keys():
            tagList = wordVsTagCount[word].keys()
            for tag in tagList:
                storeResult[i][tag] = {}
                maxValue = -sys.maxsize - 1
                result = ''
                emission = wordVsTagCount[word][tag]
                for previousTag in storeResult[i - 1].keys():
                    probablity = storeResult[i - 1][previousTag]['probablity'] + emission + tagToTagCount[previousTag][tag]
                    if probablity > maxValue:
                        maxValue = probablity
                        result = previousTag
                storeResult[i][tag] = {}
                storeResult[i][tag]['probablity'] = maxValue
                storeResult[i][tag]['backpointer'] = result
        else:
            tagList = tagVsCount.keys()
            for tag in tagList:
                if tag != 'start' and tag != 'end':  
                    storeResult[i][tag] = {}
                    maxValue = -sys.maxsize - 1
                    result = ''
                    for previousTag in storeResult[i - 1].keys():
                        probablity = storeResult[i - 1][previousTag]['probablity'] + tagToTagCount[previousTag][tag]
                        if probablity > maxValue:
                            maxValue = probablity
                            result = previousTag
                    storeResult[i][tag] = {}
                    storeResult[i][tag]['probablity'] = maxValue
                    storeResult[i][tag]['backpointer'] = result
    
    #tag the results
    taggedSentence = ""
    startTag = 'end'
    i = len(storeResult) - 1;
    j = len(storeResult) - 2;
    while i - 1 >= 0:
        tag = storeResult[i][startTag]['backpointer']
        if(i == len(storeResult) - 1):
            taggedSentence = words[j] + "/" + tag
        else:
            taggedSentence = words[j] + "/" + tag + " " + taggedSentence
        startTag = tag
        i = i - 1
        j = j - 1
    finalResult.append(taggedSentence)
    
#write to the file    
opFile = "hmmoutput.txt"
fileobj = open(opFile, mode='w', encoding='UTF-8')
for line in finalResult:
   fileobj.write(line)
   fileobj.write('\n')
fileobj.close()
