import json, sys
import math
path = sys.argv[1];
#using UTF-8 because, we can get any text format..
f = open(path, encoding = 'UTF-8');
#read from the file..
lines = f.readlines();
tagVsCount = {'start': len(lines),'end': len(lines)};

tagToTagCount = {};

wordVsTagCount = {};

known_tags = set();
known_tags.add('start')
known_tags.add('end')
for sentence in lines:
    prevTag = 'start';
    lastWord = '';
    lastTag = '';
    for word in sentence.split():
        wordVsTag = word.split("/");
        tag = wordVsTag[1];
        known_tags.add(tag);
        #increment the tag
        tagVsCount[tag] =  tagVsCount.get(tag, 0) + 1;
        
        currWord = wordVsTag[0];
        
        #increment  word -> tag -> count
        tagVsCountForGivenWord = wordVsTagCount.get(currWord, {});
        tagVsCountForGivenWord[tag] = tagVsCountForGivenWord.get(tag, 0) + 1;
        wordVsTagCount[currWord] = tagVsCountForGivenWord;
        
        #increment tag->tag -> count
        currTagCount = tagToTagCount.get(prevTag, {});
        currTagCount[tag] = currTagCount.get(tag, 0) + 1;
        tagToTagCount[prevTag] = currTagCount;
        
        lastWord = currWord;
        lastTag = tag;
        prevTag = tag;
    #increment tag->tag -> count
    currTagCount = tagToTagCount.get(lastTag, {});
    currTagCount['end'] = currTagCount.get('end', 0) + 1;
    tagToTagCount[lastTag] = currTagCount;  

for prevTag, nextTag in tagToTagCount.items():
    for tag in known_tags:
        if tag not in nextTag:
            tagToTagCount[prevTag][tag] = 0;
            
for prevTag in tagToTagCount:
    for currentTag in tagToTagCount[prevTag]:
        tagToTagCount[prevTag][currentTag] = math.log(tagToTagCount[prevTag][currentTag] + 1)  - math.log(tagVsCount[prevTag] + 1);

for word in wordVsTagCount:
    for tag in wordVsTagCount[word]:
        wordVsTagCount[word][tag] = math.log(wordVsTagCount[word][tag]) - math.log(tagVsCount[tag]);

modelFile = 'hmmmodel.txt'
result=[];
result.append(tagVsCount)
result.append(tagToTagCount)
result.append(wordVsTagCount)
fileWrite = open(modelFile, mode = 'w', encoding = 'UTF-8')
fileWrite.write(json.dumps(result))
fileWrite.close()
