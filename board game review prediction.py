#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


games=pandas.read_csv("games.csv")


# In[4]:


print games.coloumns

print games.shape


# In[5]:


print (games.columns)
print (games.shape)


# In[6]:


#hist
plt.hist(games["average_rating"])
plt.show()


# In[7]:


#print 1 row
print(games[games["average_rating"]==0].iloc[0])
print(games[games["average_rating"]>0].iloc[0])


# In[8]:


games=games[games["users_rated"]>0]

games=games.dropna(axis=0)

plt.hist(games["average_rating"])
plt.show()


# In[9]:


print(games.columns)


# In[11]:


corrmat=games.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8, square= True)
plt.show()


# In[13]:


#all col from datframe
columns=games.columns.tolist()

#filte the col to remove useless data
columns=[c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]
target ="average_rating"


# In[15]:


#gen tarin and test dataset
from sklearn.model_selection import train_test_split
train= games.sample(frac=0.8,random_state=1)
test=games.loc[~games.index.isin(train.index)]
print(train.shape)
print(test.shape)


# In[24]:


#import lin reg model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
LR=LinearRegression()
LR.fit(train[columns],train[target])


# In[26]:


#gen pred
predictions=LR.predict(test[columns])
#comute error
mean_squared_error(predictions,test[target])


# In[27]:


from sklearn.ensemble import RandomForestRegressor
#init the model
RFR=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)
RFR.fit(train[columns],train[target])


# In[30]:


predictions=RFR.predict(test[columns])
mean_squared_error(predictions,test[target])


# In[31]:


test[columns].iloc[0]


# In[33]:


rating_LR=LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR=RFR.predict(test[columns].iloc[0].values.reshape(1,-1))
print (rating_LR)
print (rating_RFR)


# In[35]:


test[target].iloc[0]


# In[ ]:




