# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:14:43 2017

@author: Bhargav Dalal
"""

import pandas as pd

data = pd.read_csv('geodata_1.csv')
data = data[data.msk == 1]
data = data[data.dymask == 1]
data = data[['t_user_id', 'date', 'transportation_mode','h_dt', 'speed', 'acc', 'bearing']]
data['date'] = pd.to_datetime(data['date'])
data = data[data.transportation_mode != 'run']
data = data[data.transportation_mode != 'motorcycle']


#Calculating 5 measures that is mean, min, max, median and standard deviation
data_Mn = data.groupby(['t_user_id', 'date', 'transportation_mode']).mean().reset_index()
data_Md = data.groupby(['t_user_id', 'date', 'transportation_mode']).median().reset_index()
data_Mnn = data.groupby(['t_user_id', 'date', 'transportation_mode']).min().reset_index()
data_Mx = data.groupby(['t_user_id', 'date', 'transportation_mode']).max().reset_index()
data_Sd = data.groupby(['t_user_id', 'date', 'transportation_mode']).std().reset_index()


data_Count = data.groupby(['t_user_id', 'date', 'transportation_mode'])[['h_dt']].count().reset_index()

#Now the column name is similar for all dataframe and hence a need arises to change the name of the parameter according to the aggregate function
#Renaming for each data frame
data_Mn.rename(columns = {'h_dt':'distance_mean','speed':'speed_mean', 'acc':'acc_mean', 'bearing':'bearing_mean'}, inplace = True)

data_Md.rename(columns = {'h_dt':'distance_median', 'speed':'speed_median', 'acc':'acc_median', 'bearing':'bearing_median'}, inplace = True)
data_Md.drop(['t_user_id', 'date', 'transportation_mode'], axis=1,inplace=True)

data_Mnn.rename(columns = {'h_dt':'distance_min', 'speed':'speed_min', 'acc':'acc_min', 'bearing':'bearing_min'}, inplace = True)
data_Mnn.drop(['t_user_id', 'date', 'transportation_mode'],axis=1,inplace=True)

data_Mx.rename(columns = {'h_dt':'distance_max', 'speed':'speed_max', 'acc':'acc_max', 'bearing':'bearing_max'}, inplace = True)
data_Mx.drop(['t_user_id', 'date', 'transportation_mode'],axis=1,inplace=True)

data_Sd.rename(columns = {'h_dt':'distance_std', 'speed':'speed_std', 'acc':'acc_std', 'bearing':'bearing_std'}, inplace = True)
data_Sd.drop(['t_user_id', 'date', 'transportation_mode'],axis=1,inplace=True)

data_Count.rename(columns = {'h_dt':'counttt'}, inplace = True)
data_Count.drop(['t_user_id', 'date', 'transportation_mode'],axis=1,inplace=True)
#Connecting all the dataframes in one data frame
clean_data = pd.concat([data_Mn, data_Mnn, data_Mx, data_Md, data_Sd, data_Count], axis = 1, join = 'inner')

#Now as we need to eliminate subtrajectories whcih are less than 10
clean_data = clean_data[clean_data.counttt > 10]
clean_data.drop(['counttt'],axis=1,inplace=True)

clean_data.to_csv('geodata_final.csv', header = True, index = False, sep=',')

