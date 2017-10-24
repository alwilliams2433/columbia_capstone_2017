
# coding: utf-8

# In[1]:

import requests
import json
import pandas as pd


# In[2]:

url = 'https://api.foursquare.com/v2/venues/search'
dictionary={}
list_of_names=[]
list_of_ids=[]
zip_codes=[10026, 10027, 10030, 10037, 10039,10001, 10011, 10018, 10019, 10020, 10036,10029, 10035,10010, 10016, 10017, 10022,10012, 10013, 10014,
           10004, 10005, 10006, 10007, 10038, 10280,10002, 10003, 10009,10021, 10028, 10044, 10065, 10075, 10128,10023, 10024, 10025,10031, 10032, 10033, 10034, 10040]
for zipcode in zip_codes:
    params = dict(
      client_id='FEIGYDUBLGCICZYQG2124RNB3M10AHZZ2SVALLZ50TE55YDB',
      client_secret='FARUQQBQKATQPWNOPFLIPPO22RWYGBN3GS5XQIRTPVOYYFFM',
      v='20170801',
    #   ne='40.73,-73.99',
    #   sw='40.68,-74.04',
      near=str(zipcode),
      #query='chinese',
      limit=5000,
      intent='browse',
      #categoryId='4bf58dd8d48988d145941735'
      categoryId='4bf58dd8d48988d145941735'
    )
    
    resp = requests.get(url=url, params=params)
    data = json.loads(resp.text)
    
    length=len(data['response']['venues'])
    
    for i in range(0,length):
        list_of_names.append(data['response']['venues'][i]['name'])
        list_of_ids.append(data['response']['venues'][i]['id'])
    dictionary[zipcode]=length




# In[3]:

#pd.DataFrame(['a']list_of_names,list_of_ids,columns=['a','b'])

chinese_restaurants=pd.DataFrame(list_of_names,columns=['name'])
chinese_restaurants['id']=list_of_ids


# In[4]:

chinese_restaurants['id']=list_of_ids


# In[ ]:

list_of_ids.length


# In[ ]:

newurl = 'https://api.foursquare.com/v2/venues/'
list_of_pricetier=[]
list_of_userscount=[]
list_of_visitscount=[]
list_of_checkinscount=[]
list_of_tipcount=[]

for ids in list_of_ids:
    
    newparams = dict(
        client_id='DK43IMDNDJZZVM23NVJGARXSSJYS4H1PH2CDLKXQBD5EPTKC',
        client_secret='2JF1PL21AR4KQFLOPTHEJMY2POHJOR0CEQPQ155JVELJPUIQ',
        v='20170801',
    )
    
    newresp = requests.get(url=newurl+ids, params=newparams)
    newdata = json.loads(newresp.text)
    
    list_of_userscount.append(newdata['response']['venue']['stats']['usersCount'])
    list_of_visitscount.append(newdata['response']['venue']['stats']['visitsCount'])
    list_of_checkinscount.append(newdata['response']['venue']['stats']['checkinsCount'])
    list_of_tipcount.append(newdata['response']['venue']['stats']['tipCount'])

    
    


# In[ ]:




# In[ ]:



