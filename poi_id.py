
# coding: utf-8

# In[ ]:


import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import numpy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from tester import test_classifier


# In[ ]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


# In[ ]:


features_list = ['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 'total_stock_value',
 'expenses',
 'loan_advances',
 'from_messages',
 'other',
 'from_this_person_to_poi',
 'director_fees',
 'deferred_income',
 'long_term_incentive',
 #'email_address',
 'from_poi_to_this_person' ,
 'long_incentive_over_salary_ratio', # new created
 'bonus_over_salary_ratio']  # new created


# In[ ]:


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[ ]:


### how many people are in the dataset
len(data_dict)


# In[ ]:


### for each person, how may features are available (include 'email_address' and 'poi', not include new created features)
len(data_dict['ALLEN PHILLIP K'])


# In[ ]:


#### how many "poi" in the dataset
poi_list =[]

for k in data_dict:
    if data_dict[k]["poi"] == True:
        poi_list.append(k)
        
print (poi_list, len(poi_list))


# In[ ]:


### features with missing vales percentage

features = data_dict['ALLEN PHILLIP K'].keys()    # get all features
features_NaN = {}

for i in range(0, len(features)):    
    feature = features[i]      # get each feature name in every for loop
    m = 0      # count how many NaN for each feature               
    for k in data_dict:      # loop each person
        if data_dict[k][feature] == 'NaN':     # if the feature is NaN for the looped person
            m += 1       # count how many people have 'NaN' for this feature
    features_NaN[feature] = round(float(m) / float(len(data_dict)), 2)       # 2 decimal percentage of NaN for each feature

features_NaN


# In[ ]:


### Task 2: Remove outliers


# Because of the size of dataset, I tend to be conservative about outliers. The two outliers I decide to remove are "TOTAL" and "THE TRAVEL AGENCY IN THE PARK", which are mentioned in the "enron61702insiderpay.pdf".

# In[ ]:


data_dict.pop('TOTAL', 0 ) 
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0 ) 


# In[ ]:


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


# It seems that POI are a group of people who chase short term benefits. So I use "bonus" as short-term benefits and "long_term_incentive" as long-term benefits, and make both of them divides by "bonus" since bonus is a fair measure of position in a company.

# In[ ]:


for person in data_dict:  # create 'long_incentive_over_salary_ratio'
    if data_dict[person]['salary'] != "NaN" and data_dict[person]['long_term_incentive'] != 'NaN':
        data_dict[person]['long_incentive_over_salary_ratio'] = float(data_dict[person]['long_term_incentive'])          / float(data_dict[person]['salary'])
    else:
        data_dict[person]['long_incentive_over_salary_ratio'] = 0
        
for person in data_dict:   # create 'bonus_over_salary_ratio'  
    if data_dict[person]['salary'] != "NaN" and data_dict[person]['bonus'] != 'NaN':
        data_dict[person]['bonus_over_salary_ratio'] = float(data_dict[person]['bonus'])          / float(data_dict[person]['salary'])
    else:
        data_dict[person]['bonus_over_salary_ratio'] = 0


# In[ ]:


# transform data_dict to data frame

df = pd.DataFrame.from_dict(data_dict)
df = df.transpose()

df.apply(lambda x: pd.to_numeric(x, errors='ignore'))

#reference https://discussions.udacity.com/t/enron-data-pandas/199298/2
 
df.replace(to_replace='NaN', value=numpy.nan, inplace=True)  
df.fillna(0, inplace = True)


# In[ ]:


# plot new created features

plt.scatter(df[df['poi']==0]['long_incentive_over_salary_ratio'],df[df['poi']== 0]['bonus_over_salary_ratio'], c = 'b')
plt.scatter(df[df['poi']==1]['long_incentive_over_salary_ratio'],df[df['poi']== 1]['bonus_over_salary_ratio'], c = 'r')
plt.legend(("non_poi", "poi"))


# From the graph, it seems that POIs are not likely having "long_incentive_over_salary_ratio" greater than 10. 

# In[ ]:


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[ ]:


### Task 4: Try a varity of classifiers & Task 5: Tune parameters


# In[ ]:


#cross validation
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
sss = StratifiedShuffleSplit(labels_train, 1000, random_state = 42)


# In[ ]:


# first classifier: GaussianNB()


kbest = SelectKBest(f_classif)
classifier = GaussianNB()

steps = [('k_best', kbest),('classifier', classifier)]

pipeline = Pipeline(steps)

param_grid = {'k_best__k': range(3, 10)}

clf = GridSearchCV(pipeline, param_grid, scoring="f1",cv=sss)
    
clf.fit(features_train, labels_train)

print clf.best_params_


# In[ ]:


# predict and scores
pred = clf.predict(features_test)
recall = recall_score(labels_test, pred)
precision = precision_score(labels_test, pred)
accuracy = accuracy_score(labels_test, pred)
print recall, precision, accuracy


# In[ ]:


# SelectKBest scores
KBest = clf.best_estimator_.named_steps['k_best']
KBest.scores_


# In[ ]:


# second classifier: SVM


scaler = MinMaxScaler()
kbest = SelectKBest(f_classif)
classifier = svm.SVC()

steps = [('scaler', scaler), ('k_best', kbest), ('classifier', classifier)]
pipeline = Pipeline(steps)

param_grid = {'k_best__k': range(10,15), 'classifier__kernel': ['linear','rbf'],
             'classifier__C':[0.01,0.1,1,10,100,1000], 'classifier__gamma':[0.001,0.01,0.1,1,10,100]}

clf = GridSearchCV(pipeline, param_grid, scoring="f1",cv=sss)
    
clf.fit(features_train, labels_train)

print clf.best_params_


# In[ ]:


pred = clf.predict(features_test)
recall = recall_score(labels_test, pred)
precision = precision_score(labels_test, pred)
accuracy = accuracy_score(labels_test, pred)
print recall, precision, accuracy


# In[ ]:


# third classifier: Decision Tree


kbest = SelectKBest(f_classif)
pca = PCA()
classifier = tree.DecisionTreeClassifier()


steps = [('k_best', kbest), ('pca', pca), ('classifier', classifier)]

pipeline = Pipeline(steps)


param_grid = {'k_best__k': range(10,15), 'pca__n_components': [2,3,4], 'classifier__min_samples_split': range(18,25),              'classifier__criterion': ['gini', 'entropy'], 'classifier__min_samples_leaf': [2,3,4,5],              'classifier__max_leaf_nodes': [None,2,3,4]}

clf = GridSearchCV(pipeline, param_grid, scoring="f1",cv=sss)
    
clf.fit(features_train, labels_train)

print clf.best_params_


# In[ ]:


pred = clf.predict(features_test)
recall = recall_score(labels_test, pred)
precision = precision_score(labels_test, pred)
accuracy = accuracy_score(labels_test, pred)
print recall, precision, accuracy


# In[ ]:


### test if the new created feature "bonus_over_salary_ratio" is really useful to improve performance of algorithm


# In[ ]:


with_new_features_list = ["poi", "bonus", "bonus_over_salary_ratio", "salary", "shared_receipt_with_poi", "total_stock_value",                       "exercised_stock_options", "total_payments", "deferred_income", "restricted_stock"]

without_new_feature_list = ["poi", "bonus", "salary", "shared_receipt_with_poi", "total_stock_value",                       "exercised_stock_options", "total_payments", "deferred_income", "restricted_stock"]


# In[ ]:


clf = GaussianNB()


# In[ ]:


# performance scores when have new created feature "bonus_over_salary_ratio" 
test_classifier(clf, my_dataset, with_new_features_list)


# In[ ]:


# performance scores when DOES NOT have new created feature "bonus_over_salary_ratio" 
test_classifier(clf, my_dataset, without_new_feature_list)


# Comparing the two test results, we found that the new created feature "bonus_over_salary_ratio" isn't helpful to improve algorithm performance. So we will delete it from the final featue list .

# In[ ]:


# Final analysis


# In[ ]:


#feature_list for final analysis
final_feature_list = ["poi", "bonus", "salary", "shared_receipt_with_poi", "total_stock_value",                       "exercised_stock_options", "total_payments", "deferred_income", "restricted_stock"]


# In[ ]:


#extract and validate data
data = featureFormat(my_dataset, final_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
sss = StratifiedShuffleSplit(labels,  1000, random_state = 42)


# In[ ]:


# final algorithm
clf = GaussianNB()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

# performance score in sklearn metrics
recall = recall_score(labels_test, pred)
precision = precision_score(labels_test, pred)

print recall, precision


# In[ ]:


# test classifier in tester.py
test_classifier(clf, my_dataset, final_feature_list)


# In[ ]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, final_feature_list)

