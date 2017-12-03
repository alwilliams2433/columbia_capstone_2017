
# coding: utf-8

# # Recommend Locations for New Business

# ## 1. Cluster the Business Venues

# ### 1.1 Import the raw check-in dataset

# In[50]:

import pandas as pd
import numpy as np
import math

checkin = pd.read_csv("nyc-checkin.csv")
checkin = checkin.drop(['timezoneOffset', 'utcTimestamp'], axis=1)
checkin


# ### 1.2 Implement k-means on all business venues

# In[51]:

venues = checkin.drop_duplicates("venueId")


# In[52]:

from sklearn.cluster import KMeans
ven_lat_lng = venues[['latitude', 'longitude']].values
k = 100
kmeans = KMeans(n_clusters=k, random_state=1).fit(ven_lat_lng)
label = kmeans.labels_
label = label.reshape(label.shape[0], 1)
cluster = np.concatenate((venues, label), axis=1)
cluster_df = pd.DataFrame({'venueId': cluster[:,1], 'latitude': cluster[:,4],'longitude': cluster[:,5], 'label':cluster[:,-1]})


# ### 1.3 Plot the clustering data

# In[53]:

#plot the scatter graph
import matplotlib.pyplot as pyplot
pyplot.scatter(cluster_df['latitude'].values, cluster_df['longitude'].values, c = cluster_df['label'],s=0.1)
# pyplot.show()


# ### 1.4 Add clustering result to dataset

# In[54]:

#Save cluster result to a csv file
cluster_df.to_csv("clustering_neighborhood.csv")
#create a map from venueId to cluster result
ven_label = cluster_df.set_index('venueId').T.to_dict('label')
cluster_id = []
for index, row in checkin.iterrows():
    cluster_id.append(ven_label[row['venueId']][0])


# In[55]:

checkin["neighborhood_id"] = cluster_id
checkin.to_csv("checkin_neighborhood.csv")
checkin


# ## 2. Pick up neighborhood using Collaborative Neighborhood Filtering (CNF)

# ### 2.1 Preparation, Construct Neighborhood-Venues Matrix

# In[56]:

"""
convert dataset to dictionary:
N_CAT matrix and GA_MAT
first transform dataset to certain format

neighbor_cate_checkin_dic:
{   
 neighborhood_id: {
     venueCategoryId: count
 }   
}

neighbor_cate_exist_dic:
{   
 neighborhood_id: {
     venueCategoryId
 }   
}

"""
# Several Tune Parameters:
# Number of Similar Neighborhoods
Sim = 5
# predic_category = "Chinese Restaurant"
top_n = 5


# Map from neighborhood to category checkin number
neighbor_cate_checkin_dic = {}
# Map from neighborhood to category existence
neighbor_cate_exist_dic = {}

for index, row in checkin.iterrows():
    neighbor = row["neighborhood_id"]
    if neighbor not in neighbor_cate_checkin_dic:
        neighbor_cate_checkin_dic[neighbor] = {}
        neighbor_cate_exist_dic[neighbor] = set()
    category = row["venueCategoryId"]
    if category not in neighbor_cate_checkin_dic[neighbor]:
        neighbor_cate_checkin_dic[neighbor][category] = 1
        neighbor_cate_exist_dic[neighbor].add(category)
    else:
        neighbor_cate_checkin_dic[neighbor][category] += 1


# In[57]:

# Map for categoryid and category
categoryid_category = checkin[['venueCategoryId', 'venueCategory']].drop_duplicates("venueCategoryId")
categoryid_category_dic = categoryid_category.set_index("venueCategoryId").T.to_dict("list")
category_categoryid_dic = categoryid_category.set_index('venueCategory').T.to_dict("list")

# Map from neighborhood to category count
neighbor_cate_count_dic = {}
for index, row in checkin[['neighborhood_id', 'venueCategory', 'venueCategoryId']].drop_duplicates("venueCategoryId").iterrows():
    n = row['neighborhood_id']
    if n not in neighbor_cate_count_dic:
        neighbor_cate_count_dic[n] = {}
    c = row['venueCategory']
    if c not in neighbor_cate_count_dic[n]:
        neighbor_cate_count_dic[n][c] = 1
    else:
        neighbor_cate_count_dic[n][c] += 1


# In[58]:

'''
Construct N_CAT matrix
First get a map from business venues to business categories.

venue_category_dic format:
{

    venuesId : { venueCategoryId: ""}
}
'''
venue_category = checkin[['venueId', 'venueCategoryId']].drop_duplicates('venueId')
venue_category_dic = venue_category.set_index("venueId").T.to_dict("list")


# In[59]:

# create a empty N_CAT Matrix
k = 100 # Number of cluster
n_cat = pd.DataFrame(0, index=range(k), columns=categoryid_category['venueCategory'])
# Map from categoriy to column number
cate_col_dic = {}
count = 0
for c in categoryid_category['venueCategory']:
    cate_col_dic[c] = count
    count += 1


# In[60]:

# Fill the N_CAT Matrix
for index, row in n_cat.iterrows():
    if index in neighbor_cate_exist_dic:
        for cate_id in neighbor_cate_exist_dic[index]:
            category = categoryid_category_dic[cate_id]
            n_cat.ix[index, category] = 1
n_cat


# ### 2.2 Find similar Neighborhoods for each Neighborhood (using N_CAT matrix)

# #### 2.2.1 Define distance metrics for similarity

# In[61]:

def euclidean_distance(index):
    return np.linalg.norm(n_cat.values[index] - n_cat.values, axis=1)

def manhattan_distance(index):
    return np.linalg.norm(n_cat.values[index] - n_cat.values, ord=1, axis=1)

def jaccard_index(index, category):
    temp = n_cat.drop(category, axis=1).values
    result = []
    for row in temp:
        dividor = 1.0 * np.sum(np.logical_and(temp[index], row))
        if dividor == 0:
            result.append(0)
        else:
            jaccard = 1.0 * np.sum(np.logical_and(temp[index], row)) / np.sum(np.logical_or(temp[index], row))
            result.append(jaccard)
            
    return np.array(result)


# #### 2.2.2 Define GA_MAT(likelihood) and Total Likelihood

# In[62]:

# Different GA_MAT value calculation Functions

# Use check-in in this category / total check-in in this neighborhood
def cal_likelihood(neighborhood, category):
    try:
        checkin = neighbor_cate_checkin_dic[neighborhood][category_categoryid_dic[category][0]]
    except:
        return 0
    total_checkin = 0
    for cat in neighbor_cate_checkin_dic[neighborhood]:
        total_checkin += neighbor_cate_checkin_dic[neighborhood][cat]
    return 1.0 * checkin / total_checkin

# Use check-in number in this category / total check-in in new york
def cal_likelihood_1(neighborhood, category):
    checkin = neighbor_cate_checkin_dic[neighborhood][category]
    total_checkin = len(checkin)
    return 1.0 * checkin / total_checkin

# Use the number of business venues of this category / total number of venues in this neighborhood.

def cal_likelihood_2(neighborhood, category):
    return neighbor_cate_count_dic[neighborhood][category] / sum(neighbor_cate_count_dic[neighborhood].values())


# In[63]:

# Function to calculate Total Likelihood in Equation 6
def L_function(similar_dic, category):
    denominator = sum(similar_dic.values())
    numerator = 0
    for sim in similar_dic:
        likelihood = cal_likelihood(sim, category)
        numerator += similar_dic[sim] * likelihood
    return 1.0 * numerator / denominator


# #### 2.2.3 Calculate similarities between similar neighborhoods

# In[64]:

# Map from neighborhood to similar neighborhood and (jaccard) distance
similar_neighbor = {}

output = pd.DataFrame(index=range(5))

def predict(predic_category):

    for n, row in n_cat.iterrows():
        if n not in similar_neighbor:
            similar_neighbor[n] = {}
        # result is jaccard_index to each neiborhood
        result = jaccard_index(n, predic_category)
        sim_group = result.argsort()[::-1][:Sim]
        dis_group = result[sim_group]
        for i in range(len(sim_group)):
            similar_neighbor[n][sim_group[i]] = dis_group[i]
        


    # #### 2.4 Calculate L function for each neighborhood and give result

    # In[65]:

    L_dic = []
    for neighbor, row in n_cat.iterrows():
        L_dic.append(L_function(similar_neighbor[neighbor], predic_category))


    # In[66]:

    # Top n neighborhoods with highest L value
    neighbor_result = np.array(L_dic).argsort()[::-1][:top_n]


    # In[67]:

    location_map = kmeans.cluster_centers_
    print(location_map[neighbor_result])

    location_predic = location_map[neighbor_result]
    location_string = []
    for point in location_predic:
        location_string.append(str(point[0]) + ':' + str(point[1]))

    output[predic_category] = np.array(location_string)

prediction = checkin.drop_duplicates("venueCategory")

for index, row in prediction.iterrows(): 
    print(str(row['venueCategory']))
    predict(str(row['venueCategory']))


outputT = output.transpose()
outputT.to_csv("result_for_each_type.csv")
