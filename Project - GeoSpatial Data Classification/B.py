# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:16:15 2017

@author: dalal
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()




data = pd.read_csv('geodata_final.csv')
X = data.drop(['t_user_id','date', 'transportation_mode'],axis =1)

target = np.array(data['transportation_mode'])

#Root node is transportation mode
#It has 2 nodes walk and others - Layer 1
target_walk = [1 if target[i] == 'walk' else 0 for i in range(len(target))]

#others is further divided in 2 nodes - ontrack(train,subway) & onroad(bus,road,taxi) - Layer 2
target_ontrack = [1 if (target[i] == 'train' or target[i] == 'subway') else 0 for i in range(len(target))]
target_onroad = [1 if (target[i] == 'taxi' or target[i] == 'car' or target[i] == 'bus') else 0 for i in range(len(target))]

#on track is furter divided on 2 final nodes - Layer 3
target_train = [1 if target[i] == 'train' else 0 for i in range(len(target))]
target_subway = [1 if target[i] == 'subway' else 0 for i in range(len(target))]

#Now for subnodes for onroad - Layer 3
target_car = [1 if target[i] == 'car' else 0 for i in range(len(target))]
target_taxi = [1 if target[i] == 'taxi' else 0 for i in range(len(target))]
target_bus = [1 if target[i] == 'bus' else 0 for i in range(len(target))]



#Calculation for hierarchical structure starts
#------------------------------------------------------------------------------
#Random Forest
#------------------------------------------------------------------------------

#layer 1 of tree
walk_arrmean = cross_val_score(rfc, X, target_walk, cv = 10, scoring = 'accuracy')

#Layer 2 of tree
ontrack_arrmean = cross_val_score(rfc, X, target_ontrack, cv = 10, scoring = 'accuracy')
onroad_arrmean = cross_val_score(rfc, X, target_onroad, cv = 10, scoring = 'accuracy')
#Layer 3
train_arrmean = cross_val_score(rfc, X, target_train, cv = 10, scoring = 'accuracy')
subway_arrmean = cross_val_score(rfc, X, target_subway, cv = 10, scoring = 'accuracy')

#Layer 3 for onroad node
car_arrmean = cross_val_score(rfc, X, target_car, cv = 10, scoring = 'accuracy')
taxi_arrmean = cross_val_score(rfc, X, target_taxi, cv = 10, scoring = 'accuracy')
bus_arrmean = cross_val_score(rfc, X, target_bus, cv = 10, scoring = 'accuracy')


rfc_h_arr = np.concatenate((walk_arrmean,ontrack_arrmean,onroad_arrmean,car_arrmean,taxi_arrmean,bus_arrmean,subway_arrmean,train_arrmean), axis = 0)
rfc_h_mean = (np.array(rfc_h_arr).mean()) * 100

print ('The accuracy of rainforest on the hierachal structure is %.2f' %(rfc_h_mean) + '%')
print ('')


#For flat structure
  
rfc_fl = cross_val_score(rfc, X, target, cv = 10, scoring = 'accuracy')
rfc_fl_mean = (np.array(rfc_fl).mean()) * 100

print ('The accuracy of rainforest on the flat structure is %.2f' %(rfc_fl_mean) + '%')
print ('')

#------------------------------------------------------------------------------
#Decision Tree
#------------------------------------------------------------------------------

#Layer 1 of tree
walk_arrmean = cross_val_score(dtree, X, target_walk, cv = 10, scoring = 'accuracy')

#Layer 2 of tree
ontrack_arrmean = cross_val_score(dtree, X, target_ontrack, cv = 10, scoring = 'accuracy')
onroad_arrmean = cross_val_score(dtree, X, target_onroad, cv = 10, scoring = 'accuracy')
#Layer 3 for ontrack node
train_arrmean = cross_val_score(dtree, X, target_train, cv = 10, scoring = 'accuracy')
subway_arrmean = cross_val_score(dtree, X, target_subway, cv = 10, scoring = 'accuracy')



#Layer 3 for onroad node
car_arrmean = cross_val_score(dtree, X, target_car, cv = 10, scoring = 'accuracy')
taxi_arrmean = cross_val_score(dtree, X, target_taxi, cv = 10, scoring = 'accuracy')
bus_arrmean = cross_val_score(dtree, X, target_bus, cv = 10, scoring = 'accuracy')


dtree_h_arr = np.concatenate((walk_arrmean,ontrack_arrmean,onroad_arrmean,car_arrmean,taxi_arrmean,bus_arrmean,subway_arrmean,train_arrmean), axis = 0)
dtree_h_mean = (np.array(dtree_h_arr).mean()) * 100

print ('The accuracy of Decision tree on the hierachal structure is %.2f' %(rfc_h_mean) + '%')
print ('')


#For flat structure
dtree_fl = cross_val_score(dtree, X, target, cv = 10, scoring = 'accuracy')

dtree_fl_mean = (np.array(dtree_fl).mean()) * 100

print ('The accuracy of decision tree on the flat structure is %.2f' %(dtree_fl_mean) + '%')



#-----------------------------------------------------------------------------
#T-TEST
#-----------------------------------------------------------------------------
from scipy import stats
alpha=0.05

t_fl, p_fl = stats.ttest_rel(rfc_fl,dtree_fl)

print("ttest between RandomForest Classification and decision tree for flat structure: t_fl = %g  p_fl = %g" % (t_fl, np.abs(p_fl)))
if (p_fl<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")


t_h, p_h = stats.ttest_rel(rfc_h_arr,dtree_h_arr)

print("ttest between RandomForest Classification and decision tree for huerarchical structure: t_h = %g  p_h = %g" % (t_h, np.abs(p_h)))
if (p_h<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")

df = pd.DataFrame(columns=['Classification Model','Accuracy'])
df.loc[-1] = ["RandomForest_h", rfc_h_mean]
df.index = df.index + 1  # shifting index
df = df.sort_index() 
df.loc[-1] = ["DecisionTree_h", dtree_h_mean]
df.index = df.index + 1  # shifting index
df = df.sort_index() 
df.loc[-1] = ["RandomForest_fl", rfc_fl_mean]
df.index = df.index + 1  # shifting index
df = df.sort_index() 
df.loc[-1] = ["DecisionTree_fl", dtree_fl_mean]
df.index = df.index + 1  # shifting index
df = df.sort_index() 
df.plot.bar('Classification Model','Accuracy')
