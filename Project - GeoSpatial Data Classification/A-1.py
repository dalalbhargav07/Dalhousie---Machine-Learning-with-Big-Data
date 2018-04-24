# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:09:02 2017

@author: User
"""

import numpy as np
import pandas as pd
import math
import datetime
import gpxpy.geo as gp
'''
def havrsine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000
    return m
'''

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing



def usr_class_seperator(df):
    #usr = df.t_user_id.unique()
    usr_list = np.array(df.t_user_id)
    msk = [0 if i == (len(usr_list) - 1) or usr_list[i] != usr_list[i+1] else 1 for i in range(len(usr_list))]
    return msk

def day_class_seperator(df):
    dy_list = np.array(df.date)
    dymask = [0 if i == (len(dy_list) - 1) or dy_list[i] != dy_list[i+1] else 1 for i in range(len(dy_list))]
    return dymask

def time_diff(df):
    df['Tdelta'] = [(df.collected_time[i+1] - df.collected_time[i]).total_seconds() if i != (len(df.t_user_id) - 1) else 0 for i in range(len(df.t_user_id))]
    df['Tdelta'] = [df.Tdelta[i] * df.msk[i] * df.dymask[i] for i in range(len(df.collected_time))]
    #df['Tdelta'] = [df.Tdelta[i] * df.dymask[i] for i in range(len(df.collected_time))]
    #df['Tdelta'] = tdelta
    return df

def compute_distance(df):
    df['h_dt'] = [gp.haversine_distance(df.latitude[i], df.longitude[i], df.latitude[i+1], df.longitude[i +1]) if i != (len(df.t_user_id) - 1) else 0 for i in range(len(df.t_user_id))]
    df['h_dt'] = [df.h_dt[i] * df.msk[i] * df.dymask[i] for i in range(len(df.t_user_id))]
    #df['h_dt'] = [df.h_dt[i] * df.dymask[i] for i in range(len(df.t_user_id))]
    #df['Haversine_distance'] = distance
    return df

def compute_speed(df):
    df['speed'] = [df.h_dt[i] / (df.Tdelta[i]+ 0.1**10) if i != (len(df.collected_time) - 1) else 0 for i in range(len(df.collected_time))]
    df['speed'] = [df.speed[i] * df.msk[i] * df.dymask[i] for i in range(len(df.h_dt))]
    #df['speed'] = [df.speed[i] * df.dymask[i] for i in range(len(df.h_dt))]
    return df

def compute_acc(df):
    df['acc'] = [df.speed[i] / (df.Tdelta[i]+ 0.1**10) if i != (len(df.collected_time) - 1) else 0 for i in range(len(df.collected_time))]
    df['acc'] = [df.acc[i] * df.msk[i] * df.dymask[i] for i in range(len(df.speed))]
    #df['acc'] = [df.acc[i] * df.dymask[i] for i in range(len(df.speed))]
    #df['Acceleration'] = acc
    return df

def set_bearing(df): 
    df['bearing'] = [calculate_initial_compass_bearing(df.Tuple[i], df.Tuple[i+1]) * df.msk[i] * df.dymask[i] if i != (len(df.t_user_id) - 1) else 0 for i in range(len(df.t_user_id))]



print (datetime.datetime.now())
data = pd.read_csv('geolife_raw.csv')
data['collected_time'] = pd.to_datetime(data['collected_time'])
data['date'] = [d.date() for d in data['collected_time']]
data['date'] = pd.to_datetime(data['date'])
data['time'] = [d.time() for d in data['collected_time']]
#data['longitude'] = np.radians(data['longitude'])
#data['latitude'] = np.radians(data['latitude'])
print (datetime.datetime.now())
data['msk'] = usr_class_seperator(data)
data['dymask'] = day_class_seperator(data)
print (datetime.datetime.now())
#Computing time Difference
time_diff(data)
print ('Time Calculation Done')
print (datetime.datetime.now())

#Computing Haversine Distance
compute_distance(data)
print ('Distance Calculation Done')
print (datetime.datetime.now())


#Computing Speed in m/s
compute_speed(data)
print ('Speed Calculation Done')
print (datetime.datetime.now())
#Computing acceleration in m/s**2
compute_acc(data)
print ('Accelearation Calculation Done')
print (datetime.datetime.now())
#Computing Bearing
data['Tuple'] = list(zip(data.latitude, data.longitude))
set_bearing(data)
print ('Bearing Calculation Done')
print (datetime.datetime.now())

data.drop(['longitude','latitude','collected_time','time'], axis = 1, inplace=True)
print ('Started writing to a new file')
data.to_csv('geodata_1.csv', header = True, index = False   , sep=',')
print ('Done')


print (datetime.datetime.now())