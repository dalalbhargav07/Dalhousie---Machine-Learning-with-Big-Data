# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:11:02 2017

@author: dalal
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
import seaborn as sns


data = pd.read_csv('geodata_final.csv')

dt = data[['transportation_mode','distance_mean','distance_min', 'distance_max', 'distance_median' ]]
spd = data[['transportation_mode','speed_min', 'speed_max','speed_mean', 'speed_median']]
acc = data[['transportation_mode','acc_mean', 'acc_median', 'acc_min', 'acc_max']]
br = data[['transportation_mode','bearing_mean', 'bearing_median', 'bearing_min', 'bearing_max']]
'''
dt_mean = data.groupby(['transportation_mode']).mean().reset_index()
dt_mean.plot.bar(x=['transportation_mode'],y=['distance_mean'])
plt.show()
'''
data_ontrack = data[(data.transportation_mode == 'car') | (data.transportation_mode == 'taxi')]
#spd = spd['transportation_mode' == 'train']
sns.barplot(x='transportation_mode',y='speed_max',data=data)
#plt.bar(data['transportation_mode'],data['speed_max'])
plt.show()

dt_1 = data.pivot_table(index='transportation_mode',columns='t_user_id',values='speed_max')
dt_1 = dt_1.fillna(0)
'''
#-------------------------------------------------
#Parallel Co-ordinates graph
#-------------------------------------------------

print ('Distance')
parallel_coordinates(dt,'transportation_mode', color = ['g','m','b','r','c','k'])    
plt.show()

plt.figure()
print ('Speed')
parallel_coordinates(spd,'transportation_mode', color = ['g','m','b','r','c','y'])
plt.show()

plt.figure()    
print ('Acceleration')
parallel_coordinates(acc,'transportation_mode', color = ['g','m','b','r','c','k'])    
plt.show()

plt.figure()
print ('Bearing')
parallel_coordinates(br,'transportation_mode', color = ['g','m','b','r','c','k'])    
plt.show()
#-------------------------------------------------



'''