import os
import normalize
import json
import glob
import sys

paths=[]
class_name=[]

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

for f in all_files:
      #print(f)
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

data=[]
def_class=[]
class_count={}
total_count = 0
prior_probablity={}
vocabulary_count = 0

classVsCount={}
classVsWordCount={}
vocab_set = set()
classVsWordProbablities={}

for i in range(len(paths)):
    content = open(paths[i]).read()
    content = normalize.normalizeData(content)
    data.append(content)
    def_class.append(class_name[i])
    total_count+=1
    count = classVsCount.get(class_name[i], 0)
    classVsCount[class_name[i]] = count + 1
#print(total_count)
#print(classVsCount)
for  key,value in classVsCount.items():
    prior = value/total_count;
    prior_probablity[key] = prior
#print(prior_probablity)

for i in range(len(data)):
    words = data[i].split()
    class_ref = def_class[i]
    #this is for finding class -> {word, count}
    wordVsCount =  classVsWordCount.get(class_ref, {})
    for word in words:
        wordcount = wordVsCount.get(word, 0)
        wordVsCount[word] = wordcount + 1;
        classVsWordCount[class_ref] = wordVsCount

#for key,val in classVsWordCount.items():
#    del_keys=[]
#    for keyIn,valueIn in val.items():
#        if valueIn > 300 or valueIn < 2:
#           del_keys.append(keyIn)
#    for keys in del_keys:
#        del classVsWordCount[key][keys]

for key,val in classVsWordCount.items():
    currCount = 0
    for keyIn,valueIn in val.items():
        currCount += valueIn
        vocab_set.add(keyIn)
    classVsCount[key] = currCount
    
vocabulary_count = len(vocab_set)
#print(vocabulary_count)
#print(classVsCount)
wordVsOverallCount={}
for word in vocab_set:
    for key,val in classVsWordCount.items():
        count = wordVsOverallCount.get(word, 0)
        add = val.get(word, 0)
        count += add
        wordVsOverallCount[word] = count
    
sorted_by_value = sorted(wordVsOverallCount.items(), key=lambda kv: kv[1])

stop_words=['other', 'really', 'first',
            'been', 'their', 'some', 'more', 'what', 'got', 
            'also', 'here', 'only', 'or', 'which', 'by', 'could', 'even',
            'did', 'after', 'about', 'will', 'just', 'again', 'get',
            'if', 'up', 'us', 'out', 'an', 
            'one', 'are', 'me', 'so', 'all',
            'when', 'would', 'be', 'from', 'as', 'you', 'there', 'have',
            'stay', 'very', 'but', 'our', 'they', 'on', 'had', 'with', 
            'this', 'were', 'is', 'that', 'at', 'my', 'it', 'for',
            'hotel', 'we', 'of', 'in', 'was', 'i', 'a', 'to', 'and', 'the', 'hotels']
for key,value in sorted_by_value:
    if value < 2:
        stop_words.append(key)
vocab_set_final = [word for word in vocab_set if word not in stop_words]
alpha = 0.6
#print(len(vocab_set_final))
for key,val in classVsWordCount.items():
        wordVsProb = classVsWordProbablities.get(key,{})
        for vocabword in vocab_set_final:
                valueIn = val.get(vocabword, 0)
                probablity = (valueIn + alpha) / (classVsCount[key] + alpha * vocabulary_count)
                wordVsProb[vocabword] = probablity
        classVsWordProbablities[key] = wordVsProb

f = open("nbmodel.txt","w")
f.write(json.dumps(classVsWordProbablities))
f.close()
