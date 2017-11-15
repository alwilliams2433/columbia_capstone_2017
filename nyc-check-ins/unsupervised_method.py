
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import math
import datetime
EARTH_RADIUS = 6378.137

checkin = pd.read_csv("nyc-checkin.csv")
checkin


# In[2]:

# lat_dic and lng_dic : { venueId: lat or lng }
lat_dic = checkin.set_index('venueId')["latitude"].to_dict()
lng_dic = checkin.set_index('venueId')["longitude"].to_dict()


# In[3]:

# construct user_ven dictionary and ven_user dictionary
user_ven_dic = {}
ven_user_dic = {}
for index, row in checkin.iterrows():
    user = row["userId"]
    ven = row["venueId"]
    if user not in user_ven_dic:
        user_ven_dic[user] = set()
    if ven not in ven_user_dic:
        ven_user_dic[ven] = set()
    user_ven_dic[user].add(ven)
    ven_user_dic[ven].add(user)


# In[4]:

C_set = set(checkin["venueCategory"])


# In[5]:

U_set = set(checkin["userId"])


# In[6]:

V_set = set(checkin["venueId"])


# In[7]:

#combine lat and long dic
# 
ven_to_lat_lng = {}
for ven_key in lat_dic:
    temp = [lat_dic[ven_key]]
    temp.append(lng_dic[ven_key])
    ven_to_lat_lng[ven_key] = temp


# In[ ]:

# Try to clustering the business venues.
from sklearn.cluster import DBSCAN


# Get distance from latitude and longtitude
def rad(d):
    return d * math.pi / 180.0;
def dis_calculate(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1);
    radLat2 = rad(lat2);
    a = radLat1 - radLat2;
    b = rad(lng1) - rad(lng2);

    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2),2) +
    math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b/2),2)));
    s = s * EARTH_RADIUS;
    return s * 0.621371;

# a custom function that just computes Euclidean distance
def distance(p1, p2):
    #p1 and p2 is id of venues
    alpha = 1.0
    diff1 = geodis(p1, p2)
    diff2 = socdis(p1, p2)
    return alpha * diff1 + (1 - alpha) * diff2

def geodis(p1, p2):
    lat = lat_dic[p1]
    lng = lng_dic[p2]
    return dis_calculate(lat, lng)

def socdis(p1, p2):
    set1 = ven_user_dic[p1]
    set2 = ven_user_dic[p2]
    j_index = len(set1 or set2) / len(set1 and set2)
    return 1 - j_index

alpha = 0.5
log = open("log.txt","w")
def newdistance(p1, p2):
    dis1 = dis_calculate(p1[0], p1[1], p2[0], p2[1])

    for ven_id, location in ven_to_lat_lng.items():

        if location == [p1[0], p1[1]]:
            id1 = ven_id
        if location == [p2[0], p2[1]]:
            id2 = ven_id
    
    try:
        dis2 = socdis(id1, id2)
    except:
        dis2 = 0
    try:
        log.write(str(datetime.datetime.now()))
    except:
        log = open("log.txt","w")
    return alpha * dis1 + (1 - alpha) * dis2


dbs = DBSCAN(metric=newdistance)


# In[ ]:

db = dbs.fit(checkin[["latitude", "longitude"]].values)


# In[70]:

f = open("label.txt","w")
c = open("components_.txt","w")
i = open("indices_.txt","w")
labels = db.labels_
comp = db.components_
indice = db.core_sample_indices_ 
f.write(labels)
c.write(comp)
i.write(indice)




# #Generate centroids for k-means
# lat_array = checkin["latitude"].values
# lat_range = lat_array.max() - lat_array.min()
# u_lat = np.random.rand(60,1) * lat_range + lat_array.min()

# lng_array = checkin["longitude"].values
# lng_range = lng_array.max() - lng_array.min()
# u_lng = np.random.rand(60,1) * lng_range + lng_array.min()

# u = np.concatenate((u_lat, u_lng), axis=1)


# In[ ]:




# In[84]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



