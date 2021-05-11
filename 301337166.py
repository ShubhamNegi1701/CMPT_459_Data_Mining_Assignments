#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd # to load and maipulate data and for one hot encoding
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs 
from queue import Queue
from pprint import pprint
import _pickle as cPickle
import queue






# In[44]:


df = pd.read_csv("./houshold2007.csv") #read train data


# In[45]:


df.head()


# In[46]:


df_jan = df[df['Date'].apply(lambda x: x.endswith('/1/2007'))]
df_jan.head()


# In[47]:


len(df_jan)


# In[48]:


# Finding the missing data
df_jan.dtypes #tells us the data type for each column


# In[11]:


# print out the unique values in the column workclass
df_jan['Date'].unique()


# In[12]:


df_jan['Time'].unique()


# In[13]:


df_jan['Global_active_power'].unique()


# In[14]:


df_jan['Global_reactive_power'].unique()
#has missing values


# In[15]:


# print out the unique values in the column education
df_jan['Voltage'].unique()


# In[16]:


df_jan['Global_intensity'].unique()
#has missing value


# In[17]:


df_jan['Sub_metering_1'].unique()
#missing values


# In[18]:


df_jan['Sub_metering_2'].unique()
#missing values


# In[19]:


#handle data with missing value
#count the number or rows with missing values
len(df_jan.loc[ (df_jan['Global_active_power'] =='?') &
               (df_jan['Voltage'] =='?') &
          (df_jan['Global_reactive_power'] =='?') &
               (df_jan['Global_intensity'] =='?') &
                   (df_jan['Sub_metering_1'] =='?') &
                   (df_jan['Sub_metering_2'] =='?')])


# In[20]:


#handle data with missing value
#count the number or rows with missing values
df_jan_categorical_no_null = df_jan.loc[ (df_jan['Global_active_power'] !='?') &
               (df_jan['Voltage'] !='?') &
          (df_jan['Global_reactive_power'] !='?') &
               (df_jan['Global_intensity'] !='?') &
                   (df_jan['Sub_metering_1'] !='?') &
                   (df_jan['Sub_metering_2'] !='?')]


# In[21]:


len(df_jan_categorical_no_null)


# In[22]:


df_jan_categorical_no_null.isnull().sum()
#2 missing values


# In[49]:


#df_nonull = df_jan.dropna()
df_nonull = df_jan_categorical_no_null.fillna(df_jan_categorical_no_null['Sub_metering_3'].value_counts().index[0])


# In[50]:


df_nonull.isnull().sum()


# In[51]:


df_nonull.info()


# In[26]:


print(df_nonull['Global_active_power'].value_counts().count())


# In[52]:


df_nonull.dtypes


# In[53]:


#converting df to float from object
df_nonull_copy = df_nonull.copy()
df_nonull['Global_active_power'] = df_nonull['Global_active_power'].astype('float')
df_nonull['Global_reactive_power'] = df_nonull['Global_reactive_power'].astype('float')
df_nonull['Voltage'] = df_nonull['Voltage'].astype('float')
df_nonull['Global_intensity'] = df_nonull['Global_intensity'].astype('float')
df_nonull['Sub_metering_1'] = df_nonull['Sub_metering_1'].astype('float')
df_nonull['Sub_metering_2'] = df_nonull['Sub_metering_2'].astype('float')
df_nonull.describe()


# In[54]:


df_nonull.dtypes


# In[55]:


df_nonull_copy.head()


# In[31]:


# Normalizing the data
[df_nonull['Global_active_power'].update((df_nonull['Global_active_power'] - df_nonull['Global_active_power'].min()) / (df_nonull['Global_active_power'].max() - df_nonull['Global_active_power'].min())) for col in df_nonull.columns]
[df_nonull['Global_reactive_power'].update((df_nonull['Global_reactive_power'] - df_nonull['Global_reactive_power'].min()) / (df_nonull['Global_reactive_power'].max() - df_nonull['Global_reactive_power'].min())) for col in df_nonull.columns]
[df_nonull['Voltage'].update((df_nonull['Voltage'] - df_nonull['Voltage'].min()) / (df_nonull['Voltage'].max() - df_nonull['Voltage'].min())) for col in df_nonull.columns]
[df_nonull['Global_intensity'].update((df_nonull['Global_intensity'] - df_nonull['Global_intensity'].min()) / (df_nonull['Global_intensity'].max() - df_nonull['Global_intensity'].min())) for col in df_nonull.columns]
[df_nonull['Sub_metering_1'].update((df_nonull['Sub_metering_1'] - df_nonull['Sub_metering_1'].min()) / (df_nonull['Sub_metering_1'].max() - df_nonull['Sub_metering_1'].min())) for col in df_nonull.columns]
[df_nonull['Sub_metering_2'].update((df_nonull['Sub_metering_2'] - df_nonull['Sub_metering_2'].min()) / (df_nonull['Sub_metering_2'].max() - df_nonull['Sub_metering_2'].min())) for col in df_nonull.columns]
[df_nonull['Sub_metering_3'].update((df_nonull['Sub_metering_3'] - df_nonull['Sub_metering_3'].min()) / (df_nonull['Sub_metering_3'].max() - df_nonull['Sub_metering_3'].min())) for col in df_nonull.columns]
df_nonull.describe()


# In[56]:


df_nonull.drop('Date', axis=1, inplace=True)
df_nonull.drop('Time', axis=1, inplace=True)
df_nonull.head()


# In[70]:


#finding the epsilon value
from sklearn.neighbors import NearestNeighbors

#Calculating the average distance between each point in the data set and its 12 nearest neighbors
neighbors = NearestNeighbors(n_neighbors=12)
neighbors_fit = neighbors.fit(df_nonull)
distances, indices = neighbors_fit.kneighbors(df_nonull)

#sort and plot distances
distances = np.sort(distances, axis=0)

#Sort distance values by ascending value and plot
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)


# In[ ]:





# In[178]:


######################################## Custom DBSCAN #################################################


from tqdm import tqdm
from itertools import cycle, islice

class CustomDBSCAN():
    def __init__(self):
        self.core = -1
        self.border = -2

    # Find all neighbour points at a distance of epsilon
    
    def neighbour_points(self, data, point_Id, eps):
        points_list = []
        for i in range(len(data)):
            # Euclidian distance
            if np.linalg.norm([x_i - y_i for x_i, y_i in zip(data[i], data[point_Id])]) <= eps:
                points_list.append(i)
        return points_list

    # Fit data into the model
    def fit(self, data, Eps, MinPt):

         # list of core/border points
        core_list = []
        border_list = []

        # set all points as outliers initially
        pointLabel =  len(data) * [-1] 
        pointCount = []

       

        # Find the neighbours of each individual point
        for i in tqdm(range(len(data))):
            pointCount.append(self.neighbour_points(data, i, Eps))

        # Find all the core points, border points and outliers
        for i in tqdm(range(len(pointCount))):
            if (len(pointCount[i]) >= MinPt):
                pointLabel[i] = self.core
                core_list.append(i)
            else:
                border_list.append(i)

        for i in border_list:
            for j in pointCount[i]:
                if j in core_list:
                    pointLabel[i] = self.border
                    break

        # Set points to the cluster

        cluster = 0

       # using a queue for BFS to find neighbors at a dist epsilon
        for i in range(len(pointLabel)):
           
            q = queue.Queue()
            if (pointLabel[i] == self.core):
                pointLabel[i] = cluster
                for x in pointCount[i]:
                    if(pointLabel[x] == self.core):
                        q.put(x)
                        pointLabel[x] = cluster
                    elif(pointLabel[x] == self.border):
                        pointLabel[x] = cluster
                    else:
                        pointLabel[x] = -1
            
                while not q.empty():
                    neighbors = pointCount[q.get()]
                    for y in neighbors:
                        if (pointLabel[y] == self.core):
                            pointLabel[y] = cluster
                            q.put(y)
                        if (pointLabel[y] == self.border):
                            pointLabel[y] = cluster
                # change to the next cluster            
                cluster += 1  

        return pointLabel, cluster


# In[142]:


#converting df to np.array then scaling the data before fitting the model
df_nonull_to_array = df_nonull
array_X = df_nonull_to_array.to_numpy()
array_X


# In[143]:


# Scaling the data before fitting it to the DBSCAN class
def scale(X):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)

X1 = scale(array_X)

X1


# In[180]:


custom_DBSCAN = CustomDBSCAN()
point_labels, clusters = custom_DBSCAN.fit(X1, 0.10, 14)


# In[181]:


print(point_labels)


# In[182]:


from collections import Counter
Counter(point_labels)


# In[133]:


#adding the list of clusters as a column in the df
df_nonull['clusters'] = point_labels


# In[137]:


df_nonull.head()


# In[223]:


# Displaying the histogram of the frequency of clusters
n, bins, patches=plt.hist(point_labels)
plt.xlabel("Clusters")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.show()

