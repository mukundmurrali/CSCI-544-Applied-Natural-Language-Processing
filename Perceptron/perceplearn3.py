
import os
import normalize
import glob
import sys
import json
import operator
import random


paths=[]
class_name=[]
stop_words=[
            'other', 'really',
            'been', 'their', 'some', 'more', 'what', 'got', 
            'also', 'here', 'only', 'or', 'which', 'by', 'could', 'even',
            'did', 'after', 'about', 'will', 'just', 'again', 'get',
            'if', 'up', 'us', 'out', 'an', 
            'one', 'are', 'me', 'so', 'all',
            'when', 'would', 'be', 'from', 'as', 'you', 'there', 'have',
            'stay', 'very', 'but', 'our', 'they', 'on', 'had', 'with', 
            'this', 'were', 'is', 'that', 'at', 'my', 'it', 'for',
             'we', 'of', 'in', 'was', 'i', 'a', 'to', 'and', 'the',
             'hotel','hotels', 'do','can','could','them','too']

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
    
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

class_data=[]
count_mat = {}
for i in range(len(paths)):
    single_data=[]
    content = open(paths[i]).read()
    content = normalize.normalizeData(content)
    splitWords =  content.split();
    curr_class_name=class_name[i]
    if curr_class_name == 1:
        single_data.append(1)
        single_data.append(1)
    if curr_class_name == 2:
        single_data.append(1)
        single_data.append(-1)
    if curr_class_name == 3:
        single_data.append(-1)
        single_data.append(1)
    if curr_class_name == 4:
        single_data.append(-1)
        single_data.append(-1)
    wordCount = {}
    for singleWord in splitWords:
        if singleWord not in stop_words:
            count = wordCount.get(singleWord, 0)
            wordCount[singleWord] = count + 1;
            count1 = count_mat.get(singleWord, 0)
            count_mat[singleWord] = count1 + 1;
    single_data.append(wordCount);
    class_data.append(single_data);      

count_mat_final={}
for feature in count_mat:
    if count_mat[feature] > 1: #doing ntg for now..
        count_mat_final[feature] = count_mat[feature];
    
#sort the map and use this
sorted_d = sorted(count_mat_final.items(), key=operator.itemgetter(1),reverse=True)
sorted_map = dict((x, y) for x, y in sorted_d);

#form feature vector
class_data_final=[]
for data_in_class in class_data:
    single_data=[]
    class_change = {}
    single_data.append(data_in_class[0]);
    single_data.append(data_in_class[1]);
    classData = data_in_class[2];
    for key in sorted_map:
            count = classData.get(key,0)
            class_change[key] = count;
    single_data.append(class_change);
    class_data_final.append(single_data)

        
weights_True_Deceptive={}
weights_Positive_Negative={}

bias_True_Deceptive=0
bias_Positive_Negative=0

avg_weights_True_Deceptive={}
avg_weights_Positive_Negative={}

avg_bias_True_Deceptive=0
avg_bias_Positive_Negative=0

no_of_iterations = 20
c = 1

def computeActivation(vector):
   true_deceptive = 0;
   positive_negative = 0;
   result=[]
   for feature in sorted_map:
           true_deceptive += weights_True_Deceptive.get(feature,0) * vector.get(feature,0)
           positive_negative += weights_Positive_Negative.get(feature,0) * vector.get(feature,0)
   true_deceptive +=  bias_True_Deceptive;
   positive_negative += bias_Positive_Negative;
   result.append(true_deceptive)
   result.append(positive_negative)
   return result;   
 

#print(class_data_final);
random.seed(42);
random.shuffle(class_data_final);
for iter in range(no_of_iterations):
    random.shuffle(class_data_final);
    converge = 1;
    for singleData in class_data_final:
            expectedTrueDeceptive = singleData[1];
            expectedPositiveNegative = singleData[0];
            vector = singleData[2];            
            result = computeActivation(vector)
            if result[0] * expectedTrueDeceptive <= 0:
                converge = 0;
                for feature in sorted_map:
                    if feature in weights_True_Deceptive:
                        weights_True_Deceptive[feature] += expectedTrueDeceptive * vector.get(feature,0)
                        avg_weights_True_Deceptive[feature] += expectedTrueDeceptive * vector.get(feature,0) * c
                    else:
                        weights_True_Deceptive[feature] = expectedTrueDeceptive * vector.get(feature,0)
                        avg_weights_True_Deceptive[feature] = expectedTrueDeceptive * vector.get(feature,0) * c
                    
                bias_True_Deceptive += expectedTrueDeceptive
                avg_bias_True_Deceptive += expectedTrueDeceptive * c

            if  result[1] * expectedPositiveNegative <= 0:
                converge = 0;
                for feature in sorted_map:
                    if feature in weights_Positive_Negative:
                        weights_Positive_Negative[feature] += expectedPositiveNegative * vector.get(feature,0)
                        avg_weights_Positive_Negative[feature] += expectedPositiveNegative * vector.get(feature,0) * c
                    else:
                        weights_Positive_Negative[feature] = expectedPositiveNegative * vector.get(feature,0)
                        avg_weights_Positive_Negative[feature] = expectedPositiveNegative * vector.get(feature,0) * c
                    
                bias_Positive_Negative += expectedPositiveNegative
                avg_bias_Positive_Negative += expectedPositiveNegative * c

            c += 1
    if converge == 1:
        break;
    
for feature in avg_weights_True_Deceptive:
    avg_weights_True_Deceptive[feature] = weights_True_Deceptive[feature] - (avg_weights_True_Deceptive[feature] / c)

avg_bias_True_Deceptive = bias_True_Deceptive - (avg_bias_True_Deceptive / c)

for feature in avg_weights_Positive_Negative:
    avg_weights_Positive_Negative[feature] = weights_Positive_Negative[feature] - (avg_weights_Positive_Negative[feature] / c)

avg_bias_Positive_Negative = bias_Positive_Negative - (avg_bias_Positive_Negative / c)

        
vanillaObject={}
vanillaObject["weights_True_Deceptive"] = weights_True_Deceptive;
vanillaObject["bias_True_Deceptive"] = bias_True_Deceptive;
vanillaObject["weights_Positive_Negative"] = weights_Positive_Negative;
vanillaObject["bias_Positive_Negative"] = bias_Positive_Negative;

fileVanilla = open("vanillamodel.txt","w")
fileVanilla.write(json.dumps(vanillaObject))


avgObject={}
avgObject["weights_True_Deceptive"] = avg_weights_True_Deceptive;
avgObject["bias_True_Deceptive"] = avg_bias_True_Deceptive;
avgObject["weights_Positive_Negative"] = avg_weights_Positive_Negative;
avgObject["bias_Positive_Negative"] = avg_bias_Positive_Negative;

fileAvg = open("averagedmodel.txt","w")
fileAvg.write(json.dumps(avgObject))
