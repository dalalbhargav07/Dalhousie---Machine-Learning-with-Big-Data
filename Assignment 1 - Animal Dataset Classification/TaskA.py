# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:52:52 2017

@author: Bhargav Dalal
"""
#importing panda package
#import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
     #Creating object
from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

#Reading the data:
anm = pd.read_csv("animals.csv") 


#Diving the class variable into a new dataframe called "cs"

cs = pd.get_dummies(anm['class'])

#Defining Feature Variable
X = anm.drop('class', axis = 1)



#Creating Target Variable:
dy = cs['DEER']
target_cattle = cs['CATTLE']
target_elk = cs['ELK']

#Splitting the data
train_deer_X, test_deer_X, train_deer_y, test_deer_y = tts(X,dy, test_size = 0.3, random_state=101)
train_cattle_X, test_cattle_X, train_cattle_y, test_cattle_y = tts(X,target_cattle, test_size = 0.3, random_state=101)
train_elk_X, test_elk_X, train_elk_y, test_elk_y = tts(X,target_elk, test_size = 0.3, random_state=101)

#Function for fitting data
def fit_log_rec(o,train,test):
    return o.fit(train,test)

def pred_log_rec(o,Xtest):
   return o.predict(Xtest)


#-----------------------------------------------------------------------------
#Logistic Regression
#-----------------------------------------------------------------------------
 

#def computation(ytest,pred):
    
fit_log_rec(lr,train_deer_X,train_deer_y)

"""
*****************************
Predicting on test data
*****************************
"""


deer_test_pred = pred_log_rec(lr,test_deer_X)

deer_lrtest_cm = confusion_matrix(test_deer_y,deer_test_pred)
deer_lrtest_acc = accuracy_score(test_deer_y,deer_test_pred) * 100

"""
*****************************
Predicting on training data
*****************************
"""

deer_train_pred = pred_log_rec(lr,train_deer_X)

deer_lrtrain_cm = confusion_matrix(train_deer_y,deer_train_pred)

deer_lrtrain_acc = accuracy_score(train_deer_y,deer_train_pred) * 100


"""
*****************************
On Cattle
*****************************
"""

fit_log_rec(lr,train_cattle_X,train_cattle_y)
#----On Test Data----
cattle_test_pred = pred_log_rec(lr,test_cattle_X)

cattle_lrtest_cm = confusion_matrix(test_cattle_y,cattle_test_pred)
cattle_lrtest_acc = accuracy_score(test_cattle_y,cattle_test_pred) * 100



#---On Training data

cattle_train_pred = pred_log_rec(lr,train_cattle_X)

cattle_lrtrain_cm = confusion_matrix(train_cattle_y,cattle_train_pred)

cattle_lrtrain_acc = accuracy_score(train_cattle_y,cattle_train_pred) * 100




"""
*****************************
On Elk
*****************************
"""

fit_log_rec(lr,train_elk_X,train_elk_y)
#----On Test Data----
elk_test_pred = pred_log_rec(lr,test_elk_X)

elk_lrtest_cm = confusion_matrix(test_elk_y,elk_test_pred)
elk_lrtest_acc = accuracy_score(test_elk_y,elk_test_pred) * 100


#---On Training data

elk_train_pred = pred_log_rec(lr,train_elk_X)

elk_lrtrain_cm = confusion_matrix(train_elk_y,elk_train_pred)

elk_lrtrain_acc = accuracy_score(train_elk_y,elk_train_pred) * 100



#-------------ENd of Logisitic Regression

#-----------------------------------------------------------------------------
#Naiyes Baiyers
#-----------------------------------------------------------------------------

"""
*****************************
On Deer
*****************************
"""
fit_log_rec(gnb,train_deer_X,train_deer_y)
#----On Test Data----
deer_test_pred = pred_log_rec(gnb,test_deer_X)

deer_gnbtest_cm = confusion_matrix(test_deer_y,deer_test_pred)
deer_gnbtest_acc = accuracy_score(test_deer_y,deer_test_pred) * 100
#print("5")


#---On Training data

deer_train_pred = pred_log_rec(gnb,train_deer_X)

deer_gnbtrain_cm = confusion_matrix(train_deer_y,deer_train_pred)

deer_gnbtrain_acc = accuracy_score(train_deer_y,deer_train_pred) * 100



"""
*****************************
On Cattle
*****************************
"""

fit_log_rec(gnb,train_cattle_X,train_cattle_y)
#----On Test Data----
cattle_test_pred = pred_log_rec(gnb,test_cattle_X)

cattle_gnbtest_cm = confusion_matrix(test_cattle_y,cattle_test_pred)
cattle_gnbtest_acc = accuracy_score(test_cattle_y,cattle_test_pred) * 100


#---On Training data

cattle_train_pred = pred_log_rec(gnb,train_cattle_X)

cattle_gnbtrain_cm = confusion_matrix(train_cattle_y,cattle_train_pred)

cattle_gnbtrain_acc = accuracy_score(train_cattle_y,cattle_train_pred) * 100



"""
*****************************
On Elk
*****************************
"""
fit_log_rec(gnb,train_elk_X,train_elk_y)
#----On Test Data----
elk_test_pred = pred_log_rec(gnb,test_elk_X)

elk_gnbtest_cm = confusion_matrix(test_elk_y,elk_test_pred)
elk_gnbtest_acc = accuracy_score(test_elk_y,elk_test_pred) * 100
#print("5")

#---On Training data

elk_train_pred = pred_log_rec(gnb,train_elk_X)

elk_gnbtrain_cm = confusion_matrix(train_elk_y,elk_train_pred)

elk_gnbtrain_acc = accuracy_score(train_elk_y,elk_train_pred) * 100



#-------------ENd of Naive Baiyes

#-----------------------------------------------------------------------------
#Decision Tree
#-----------------------------------------------------------------------------

"""
*****************************
On Deer
*****************************
"""
fit_log_rec(dtree,train_deer_X,train_deer_y)
#----On Test Data----
deer_test_pred = pred_log_rec(dtree,test_deer_X)

deer_dtreetest_cm = confusion_matrix(test_deer_y,deer_test_pred)
deer_dtreetest_acc = accuracy_score(test_deer_y,deer_test_pred) * 100
#print("5")


#---On Training data

deer_train_pred = pred_log_rec(dtree,train_deer_X)

deer_dtreetrain_cm = confusion_matrix(train_deer_y,deer_train_pred)

deer_dtreetrain_acc = accuracy_score(train_deer_y,deer_train_pred) * 100



"""
*****************************
On Cattle
*****************************
"""

fit_log_rec(dtree,train_cattle_X,train_cattle_y)
#----On Test Data----
cattle_test_pred = pred_log_rec(dtree,test_cattle_X)

cattle_dtreetest_cm = confusion_matrix(test_cattle_y,cattle_test_pred)
cattle_dtreetest_acc = accuracy_score(test_cattle_y,cattle_test_pred) * 100


#---On Training data

cattle_train_pred = pred_log_rec(dtree,train_cattle_X)

cattle_dtreetrain_cm = confusion_matrix(train_cattle_y,cattle_train_pred)

cattle_dtreetrain_acc = accuracy_score(train_cattle_y,cattle_train_pred) * 100



"""
*****************************
On Elk
*****************************
"""
fit_log_rec(dtree,train_elk_X,train_elk_y)
#----On Test Data----
elk_test_pred = pred_log_rec(dtree,test_elk_X)

elk_dtreetest_cm = confusion_matrix(test_elk_y,elk_test_pred)
elk_dtreetest_acc = accuracy_score(test_elk_y,elk_test_pred) * 100
#print("5")

#---On Training data

elk_train_pred = pred_log_rec(dtree,train_elk_X)

elk_dtreetrain_cm = confusion_matrix(train_elk_y,elk_train_pred)

elk_dtreetrain_acc = accuracy_score(train_elk_y,elk_train_pred) * 100

#-------------ENd of Decision Tree Classufucation

#-----------------------------------------------------------------------------
#Random Forest Classification
#-----------------------------------------------------------------------------

"""
*****************************
On Deer
*****************************
"""
fit_log_rec(rfc,train_deer_X,train_deer_y)
#----On Test Data----
deer_test_pred = pred_log_rec(rfc,test_deer_X)

deer_rfctest_cm = confusion_matrix(test_deer_y,deer_test_pred)
deer_rfctest_acc = accuracy_score(test_deer_y,deer_test_pred) * 100
#print("5")


#---On Training data

deer_train_pred = pred_log_rec(rfc,train_deer_X)

deer_rfctrain_cm = confusion_matrix(train_deer_y,deer_train_pred)

deer_rfctrain_acc = accuracy_score(train_deer_y,deer_train_pred) * 100



"""
*****************************
On Cattle
*****************************
"""

fit_log_rec(rfc,train_cattle_X,train_cattle_y)
#----On Test Data----
cattle_test_pred = pred_log_rec(rfc,test_cattle_X)

cattle_rfctest_cm = confusion_matrix(test_cattle_y,cattle_test_pred)
cattle_rfctest_acc = accuracy_score(test_cattle_y,cattle_test_pred) * 100


#---On Training data

cattle_train_pred = pred_log_rec(rfc,train_cattle_X)

cattle_rfctrain_cm = confusion_matrix(train_cattle_y,cattle_train_pred)

cattle_rfctrain_acc = accuracy_score(train_cattle_y,cattle_train_pred) * 100



"""
*****************************
On Elk
*****************************
"""
fit_log_rec(rfc,train_elk_X,train_elk_y)
#----On Test Data----
elk_test_pred = pred_log_rec(rfc,test_elk_X)

elk_rfctest_cm = confusion_matrix(test_elk_y,elk_test_pred)
elk_rfctest_acc = accuracy_score(test_elk_y,elk_test_pred) * 100
#print("5")

#---On Training data

elk_train_pred = pred_log_rec(rfc,train_elk_X)

elk_rfctrain_cm = confusion_matrix(train_elk_y,elk_train_pred)

elk_rfctrain_acc = accuracy_score(train_elk_y,elk_train_pred) * 100

'''
----------------------------------------------------------------------------------
Program Code Logic End
Below code is just for printing the outputs.
----------------------------------------------------------------------------------
#deer = {Confusion_Matrix_Test: ([deer_lrtest_cm, deer_gnbtest_cm, deer_dtreetest_cm, deer_rfctest_cm], index = ['Logistic Regression', 'Naiye Baiyes', 'Decision Tree Classification','Random Forest Classification' ] )}
'''


print ('---------------------------------DEER----------------------------------')
print('')
print('D(1). Logistic Regression for Deer:')
print('')
print('Confusion Matrix on Test Data:') 
print(deer_lrtest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(deer_lrtrain_cm)

print('Accuracy on Test Data: %.2f' %(deer_lrtest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(deer_lrtrain_acc)+'%')

print('')


print('D(2). Naiyes Baiyer Classification for Deer:')
print('')
print('Confusion Matrix on Test Data:') 
print(deer_gnbtest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(deer_gnbtrain_cm)

print('Accuracy on Test Data: %.2f' %(deer_gnbtest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(deer_gnbtrain_acc)+'%')

print('')


print('D(3). Decesion Tree for Deer:')
print('')
print('Confusion Matrix on Test Data:') 
print(deer_dtreetest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(deer_dtreetrain_cm)

print('Accuracy on Test Data: %.2f' %(deer_dtreetest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(deer_dtreetrain_acc)+'%')

print('')

print('D(4). RandomForest Classification for Deer:')
print('')
print('Confusion Matrix on Test Data:') 
print(deer_rfctest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(deer_rfctrain_cm)

print('Accuracy on Test Data: %.2f' %(deer_rfctest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(deer_rfctrain_acc)+'%')

print('')


print ('---------------------------------CATTLE----------------------------------')
print('')
print('C(1). Logistic Regression for cattle:')
print('')
print('Confusion Matrix on Test Data:') 
print(cattle_lrtest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(cattle_lrtrain_cm)

print('Accuracy on Test Data: %.2f' %(cattle_lrtest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(cattle_lrtrain_acc)+'%')

print('')


print('C(2). Naiyes Baiyer Classification for cattle:')
print('')
print('Confusion Matrix on Test Data:') 
print(cattle_gnbtest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(cattle_gnbtrain_cm)

print('Accuracy on Test Data: %.2f' %(cattle_gnbtest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(cattle_gnbtrain_acc)+'%')

print('')


print('C(3). Decesion Tree for cattle:')
print('')
print('Confusion Matrix on Test Data:') 
print(cattle_dtreetest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(cattle_dtreetrain_cm)

print('Accuracy on Test Data: %.2f' %(cattle_dtreetest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(cattle_dtreetrain_acc)+'%')

print('')

print('C(4). RandomForest Classification for cattle:')
print('')
print('Confusion Matrix on Test Data:') 
print(cattle_rfctest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(cattle_rfctrain_cm)

print('Accuracy on Test Data: %.2f' %(cattle_rfctest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(cattle_rfctrain_acc)+'%')

print('')


print ('---------------------------------ELK----------------------------------')
print('')
print('E(1). Logistic Regression for elk:')
print('')
print('Confusion Matrix on Test Data:') 
print(elk_lrtest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(elk_lrtrain_cm)

print('Accuracy on Test Data: %.2f' %(elk_lrtest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(elk_lrtrain_acc)+'%')

print('')


print('E(2). Naiyes Baiyer Classification for elk:')
print('')
print('Confusion Matrix on Test Data:') 
print(elk_gnbtest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(elk_gnbtrain_cm)

print('Accuracy on Test Data: %.2f' %(elk_gnbtest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(elk_gnbtrain_acc)+'%')

print('')


print('E(3). Decesion Tree for elk:')
print('')
print('Confusion Matrix on Test Data:') 
print(elk_dtreetest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(elk_dtreetrain_cm)

print('Accuracy on Test Data: %.2f' %(elk_dtreetest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(elk_dtreetrain_acc)+'%')

print('')

print('E(4). RandomForest Classification for elk:')
print('')
print('Confusion Matrix on Test Data:') 
print(elk_rfctest_cm)
print('')
print('Confusion Matrix on Train Data:') 
print(elk_rfctrain_cm)

print('Accuracy on Test Data: %.2f' %(elk_rfctest_acc)+'%')
print('')
print('Accuracy on Train Data: %.2f' %(elk_rfctrain_acc)+'%')

print('')
