# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
df = pd.read_csv('W55B - CloudSubSection.txt')
dfArray = df[['//X','Y','Z']].values                                           #arranging data into a structured network based on attributes and then another network using xyz coordinates

from sklearn. neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np


neighbors = NearestNeighbors(n_neighbors = 35, algorithm = 'ball_tree').fit(dfArray)   #calculating distances to other points in and keeping smallest distances to get nearest neighbors
distances, indices = neighbors.kneighbors(dfArray)                                      
                                                                                        

       
def PCA(data, correlation = False, sort = True):                               #calcuating eigenvectors and eigenvalues for normal and plane calculation at each point
    mean = np.mean(data, axis=0)
    data_adjust = data - mean
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
    else:
        matrix = np.cov(data_adjust.T) 
        
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors


def best_fitting_plane(points, equation=False):                                #using eigenvector and eigenvalues to fit a plane to the neigbourhood and find normal
    w, v = PCA(points)
    normal = v[:,2]
    point = np.mean(points, axis=0)


    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d

    else:
        return w, point, normal
 

    
def normalization(n):                                                          #normalizing function for features to remove biases
    min_max_scaler = preprocessing.MinMaxScaler()                              #takes in 2D array and uses minimum and maximum values to scale array
    np_scaled = min_max_scaler.fit_transform(n)
    df_normalized = pd.DataFrame(np_scaled)                                    #returns panda dataframe of normalized data

    return df_normalized     
 

feature_lst1 = [] 
feature_lst2 = []
feature_lst3 = []
feature_lst4 = []
feature_lst5 = []
xs = []
ys = []
zs = []                                                                        #empty lists to make arranging feature values easier 
intensity = []
height = []

for i in range(len(indices)):                                                  #populating lists with values from pandas dataframe
    xs1 = df.loc[i]['//X']
    ys1 = df.loc[i]['Y']
    zs1 = df.loc[i]['Z']
    intensity_vals = df.loc[i]['Intensity']
    
    xs.append(xs1)
    ys.append(ys1)
    zs.append(zs1)
    intensity.append(intensity_vals)
    height.append(zs1)

ints = (np.array(intensity)).reshape(-1,1)                                     #converting lists to 2D array for normalization function
int_norm = normalization(ints)

hts = (np.array(height)).reshape(-1,1)    
ht_norm = normalization(hts)

for n,a in enumerate(indices):                                                 #finding list index(n) of each of the indices and the index itself(a)
    iPCA = []
    for b in a:                                                                #for each point in the neigbourhood(indices)
        iPCA.append([df.iloc[b]['//X'],df.iloc[b]['Y'],df.iloc[b]['Z']])       #takes xyz coordinate of each point in neighbourhood and makes list for PCA
    
    eivals, pt, norms = best_fitting_plane(iPCA)
    evreshape = eivals.reshape(-1,1)
    ev_norm = normalization(evreshape)                                         #converting to 2D array for normalization
    
    ht = ht_norm.iloc[n][0]
    intens = int_norm.iloc[n][0]                                               #normalized height and intensity features for each point at the n index in pandas dataframe
#    evr2 = (np.array(ev_norm)).reshape(1,-1)
    ev = ev_norm[0][0],ev_norm[0][1],ev_norm[0][2]
    norml = norms.tolist()
    normal = norml[0],norml[1],norml[2]
   
    feature_lst1.append([ht, intens, normal])
    feature_lst2.append([])
    feature_lst3.append([])
    feature_lst4.append([])                                                    #list of feature values for k means clustering
    feature_lst4.append([])
    

   
kmeans1 = KMeans(n_clusters=10, random_state=0).fit(feature_lst1)
kmeans_labels1 = kmeans1.labels
labels1 = {'X': xs, 'Y': ys, 'Z': zs, 'kmeans labels': kmeans_labels1}
labelled1 = pd.DataFrame(data = labels1)
labelled1.to_csv('classified1.csv')

kmeans2 = KMeans(n_clusters=10, random_state=0).fit(feature_lst2)
kmeans_labels2 = kmeans2.labels
labels2 = {'X': xs, 'Y': ys, 'Z': zs, 'kmeans labels': kmeans_labels2}
labelled2 = pd.DataFrame(data = labels2)
labelled2.to_csv('classified2.csv')

kmeans3 = KMeans(n_clusters=10, random_state=0).fit(feature_lst3)
kmeans_labels3 = kmeans3.labels
labels3 = {'X': xs, 'Y': ys, 'Z': zs, 'kmeans labels': kmeans_labels3}
labelled3 = pd.DataFrame(data = labels3)
labelled3.to_csv('classified3.csv')

kmeans4 = KMeans(n_clusters=10, random_state=0).fit(feature_lst4)
kmeans_labels4 = kmeans4.labels
labels4 = {'X': xs, 'Y': ys, 'Z': zs, 'kmeans labels': kmeans_labels4}
labelled4 = pd.DataFrame(data = labels4)
labelled4.to_csv('classified4.csv')

kmeans5 = KMeans(n_clusters=10, random_state=0).fit(feature_lst5)
kmeans_labels5 = kmeans5.labels
labels5 = {'X': xs, 'Y': ys, 'Z': zs, 'kmeans labels': kmeans_labels5}
labelled5 = pd.DataFrame(data = labels5)
labelled5.to_csv('classified5.csv')


