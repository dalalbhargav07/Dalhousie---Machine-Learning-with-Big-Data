# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:43:40 2017

@author: Bhargav Dalal
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold 
#from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
     #Creating object

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc_20 = RandomForestClassifier(n_estimators = 20)
rfc_50 = RandomForestClassifier(n_estimators = 50)
rfc_100 = RandomForestClassifier(n_estimators = 100)



data = pd.read_csv("animals.csv")

X = data.drop('class', axis = 1)

cs = pd.get_dummies(data['class'])

target_deer = cs['DEER']
target_cattle = cs['CATTLE']
target_elk = cs['ELK']

#Splitting of data into testing and training
kf = KFold(len(X),10,shuffle = True)

#-----------------------------------------------------------------------------
#Random Forest Classification
#-----------------------------------------------------------------------------


#For Deer
deer_arrmean = cross_val_score(rfc, X, target_deer, cv = 10, scoring = 'accuracy')
deer_arrmean_20 = cross_val_score(rfc_20, X, target_deer, cv = 10, scoring = 'accuracy')
deer_arrmean_50 = cross_val_score(rfc_50, X, target_deer, cv = 10, scoring = 'accuracy')
deer_arrmean_100 = cross_val_score(rfc_100, X, target_deer, cv = 10, scoring = 'accuracy')

#For Cattle
cattle_arrmean = cross_val_score(rfc, X, target_cattle, cv = 10, scoring = 'accuracy')
cattle_arrmean_20 = cross_val_score(rfc_20, X, target_cattle, cv = 10, scoring = 'accuracy')
cattle_arrmean_50 = cross_val_score(rfc_50, X, target_cattle, cv = 10, scoring = 'accuracy')
cattle_arrmean_100 = cross_val_score(rfc_100, X, target_cattle, cv = 10, scoring = 'accuracy')


#For elk
elk_arrmean = cross_val_score(rfc, X, target_elk, cv = 10, scoring = 'accuracy')
elk_arrmean_20 = cross_val_score(rfc_20, X, target_elk, cv = 10, scoring = 'accuracy')
elk_arrmean_50 = cross_val_score(rfc_50, X, target_elk, cv = 10, scoring = 'accuracy')
elk_arrmean_100 = cross_val_score(rfc_100, X, target_elk, cv = 10, scoring = 'accuracy')


#Calculating Mean & Standard Deviation
#rfc_arrmean = []
rfc_arrmean = np.concatenate((deer_arrmean, cattle_arrmean, elk_arrmean), axis = 0)
rfc_mean = (np.array(rfc_arrmean).mean()) * 100
print ('Accuracy for Random Forest Classification 10 is: %.2f' %(rfc_mean) + '%')


rfc_arrmean_20 = np.concatenate((deer_arrmean_20, cattle_arrmean_20, elk_arrmean_20), axis = 0)
rfc_mean_20 = (np.array(rfc_arrmean_20).mean()) * 100
print ('Accuracy for Random Forest Classification 20 is: %.2f' %(rfc_mean_20) + '%')



rfc_arrmean_50 = np.concatenate((deer_arrmean_50, cattle_arrmean_50, elk_arrmean_50), axis = 0)
rfc_mean_50 = (np.array(rfc_arrmean_50).mean()) * 100
print ('Accuracy for Random Forest Classification 50 is: %.2f' %(rfc_mean_50) + '%')

rfc_arrmean_100 = np.concatenate((deer_arrmean_100, cattle_arrmean_100, elk_arrmean_100), axis = 0)

rfc_mean_100 = (np.array(rfc_arrmean_100).mean()) * 100
print ('Accuracy for Random Forest Classification 100 is: %.2f' %(rfc_mean_100) + '%')


#-----------------------------------------------------------------------------
#Logistic Regression
#-----------------------------------------------------------------------------
 
#For Deer
deer_arrmean = cross_val_score(lr, X, target_deer, cv = 10, scoring = 'accuracy')

#For Cattle
cattle_arrmean = cross_val_score(lr, X, target_cattle, cv = 10, scoring = 'accuracy')


#For elk
elk_arrmean = cross_val_score(lr, X, target_elk, cv = 10, scoring = 'accuracy')

#Calculating Mean & Standard Deviation
main_lr_arrmean = []
main_lr_arrmean = np.concatenate((deer_arrmean, cattle_arrmean, elk_arrmean), axis = 0)
lr_mean = (np.array(main_lr_arrmean).mean()) 
lr_std = (np.array(main_lr_arrmean).std())
print ('Accuracy for Logisitic Regression is: %.3f' % (lr_mean * 100) + '%')
print ('Standard Deviation for Logisitic Regression Classification is: %.3f' %(lr_std))



#-----------------------------------------------------------------------------
#Naive Baiyes Classification
#-----------------------------------------------------------------------------

#For Deer
deer_arrmean = cross_val_score(gnb, X, target_deer, cv = 10, scoring = 'accuracy')


#For Cattle
cattle_arrmean = cross_val_score(gnb, X, target_cattle, cv = 10, scoring = 'accuracy')


#For elk
elk_arrmean = cross_val_score(gnb, X, target_elk, cv = 10, scoring = 'accuracy')

#Calculating Mean & Standard Deviation
main_gnb_arrmean = []
main_gnb_arrmean = np.concatenate((deer_arrmean, cattle_arrmean, elk_arrmean), axis = 0)
gnb_mean = (np.array(main_gnb_arrmean).mean()) 
gnb_std = (np.array(main_gnb_arrmean).std())
print ('Accuracy for Naive Baiyes Classification is: %.3f' % (gnb_mean * 100) + '%')
print ('Standard Deviation for Naive Baiyes Classification is: %.3f' %(gnb_std ))

#-----------------------------------------------------------------------------
#Decision Tree Classification
#-----------------------------------------------------------------------------

#For Deer
deer_arrmean = cross_val_score(dtree, X, target_deer, cv = 10, scoring = 'accuracy')

#For Cattle
cattle_arrmean = cross_val_score(dtree, X, target_cattle, cv = 10, scoring = 'accuracy')


#For elk
elk_arrmean = cross_val_score(dtree, X, target_elk, cv = 10, scoring = 'accuracy')

#Calculating Mean & Standard Deviation
main_dtree_arrmean = []
main_dtree_arrmean = np.concatenate((deer_arrmean, cattle_arrmean, elk_arrmean), axis = 0)
dtree_mean = (np.array(main_dtree_arrmean).mean()) 
dtree_std = (np.array(main_dtree_arrmean).std())
print ('Accuracy for Decision Tree Classification is: %.3f' % (dtree_mean * 100) + '%')
print ('Standard Deviation for Decision Tree Classification is: %.3f' %(dtree_std))


#Student's t-test 

#-----------------------------------------------------------------------------
#Logisitc Regression
#-----------------------------------------------------------------------------

from scipy import stats
alpha=0.05

t_lr, p_lr = stats.ttest_rel(rfc_arrmean_50, main_lr_arrmean)

print("ttest between RandomForest Classification and Logistic Regression: t_lr = %g  p_lr = %g" % (t_lr, np.abs(p_lr)))
if (p_lr<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")

#-----------------------------------------------------------------------------
#Naive Baiyes Classification
#-----------------------------------------------------------------------------

t_gnb, p_gnb = stats.ttest_rel(rfc_arrmean_50, main_gnb_arrmean)

print("ttest between RandomForest Classification and Naive Baiyes: t_gnb = %g  p_gnb = %g" % (t_gnb, p_gnb))
if (p_gnb<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")


#-----------------------------------------------------------------------------
#Decision Tree Classification
#-----------------------------------------------------------------------------

t_dtree, p_dtree = stats.ttest_rel(rfc_arrmean_50, main_dtree_arrmean)

print("ttest between RandomForest Classification and Decision Tree: t_dtree = %g  p_dtree = %g" % (t_dtree, p_dtree))
if (p_dtree<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")

