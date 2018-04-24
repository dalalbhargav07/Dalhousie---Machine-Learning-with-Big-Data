# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:52:52 2017

@author: Bhargav Dalal
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold 
from sklearn.metrics import accuracy_score
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



data = pd.read_csv("animals.csv")

X = data.drop('class', axis = 1)

cs = pd.get_dummies(data['class'])

target_deer = cs['DEER']
target_cattle = cs['CATTLE']
target_elk = cs['ELK']

#Splitting of data into testing and training
kf = KFold(len(X),10,shuffle = True)
def acc(y,pred):
    return accuracy_score(y,pred)

#deer_lr_arrmean = []
'''
for train_index, test_index in kf:
    #print("Train INdex:", train_index,"test index:",test_index)
    
    #train_X_deer, test_X_deer = X[train_index], X[test_index]
    #train_y_deer, test_y_deer = y_target_deer[train_index], y_target_deer[test_index]
    lr.fit(X[train_index], target_deer[train_index])
    probas = lr.predict(X[testt_index])
    deer_mean = acc(target_deer[test_index], probas) 
    cattle_mean = acc(target_cattle[test_index], probas)
    deer_lr_arrmean.append(deer_mean)
deer_lr_mean = np.array(deer_lr_arrmean).mean()
print ("Mean: ", deer_lr_mean)
print (deer_lr_arrmean)

'''


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


#-----------------------------------------------------------------------------
#Random Forest Classification
#-----------------------------------------------------------------------------


#For Deer
deer_arrmean = cross_val_score(rfc, X, target_deer, cv = 10, scoring = 'accuracy')

#For Cattle
cattle_arrmean = cross_val_score(rfc, X, target_cattle, cv = 10, scoring = 'accuracy')


#For elk
elk_arrmean = cross_val_score(rfc, X, target_elk, cv = 10, scoring = 'accuracy')

#Calculating Mean & Standard Deviation
main_rfc_arrmean = []
main_rfc_arrmean = np.concatenate((deer_arrmean, cattle_arrmean, elk_arrmean), axis = 0)
rfc_mean = (np.array(main_rfc_arrmean).mean())
rfc_std = (np.array(main_lr_arrmean).std())
print ('Accuracy for Random Forest Classification is: %.3f' %(rfc_mean * 100 ) + '%')
print ('Standard Deviation for Random Forest Classification is: %.3f' %(rfc_std))


#Student's t-test 

#-----------------------------------------------------------------------------
#Logisitc Regression
#-----------------------------------------------------------------------------

from scipy import stats
alpha=0.05

t_lr, p_lr = stats.ttest_rel(main_rfc_arrmean, main_lr_arrmean)

print("ttest between RandomForest Classification and Logistic Regression: t_lr = %g  p_lr = %g" % (t_lr, np.abs(p_lr)))
if (p_lr<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")

#-----------------------------------------------------------------------------
#Naive Baiyes Classification
#-----------------------------------------------------------------------------

t_gnb, p_gnb = stats.ttest_rel(main_rfc_arrmean, main_gnb_arrmean)

print("ttest between RandomForest Classification and Naive Baiyes: t_gnb = %g  p_gnb = %g" % (t_gnb, p_gnb))
if (p_gnb<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")


#-----------------------------------------------------------------------------
#Decision Tree Classification
#-----------------------------------------------------------------------------

t_dtree, p_dtree = stats.ttest_rel(main_rfc_arrmean, main_dtree_arrmean)

print("ttest between RandomForest Classification and Decision Tree: t_dtree = %g  p_dtree = %g" % (t_dtree, p_dtree))
if (p_dtree<alpha):
    print("It is statistically significant")
else:
    print("It is not statistically significant")