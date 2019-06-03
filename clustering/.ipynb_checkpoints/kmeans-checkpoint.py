import pandas as pd
import _pickle as pickle
import numpy as np
import time
import math
import random

# def processing(X_df):
#     #remove the 
#     #make the array list same shapes
#     max_len = 31
#     rst = []
#     data = filter(lambda x:len(x)<31, data)
#     X_df = X_df[len(X_df.sgvs)<=31]
#     padded = []
#     for bgs in data:
#         len_bgs = len(bgs)
#         if len_bgs < max_len:
#             # padding with mean value
#             value = int(np.sum(bgs)/len_bgs)
#             paddings = [value for x in range(len_bgs, max_len)]
#             # interps = np.interp([x for x in range(len_bgs, max_len)], [x for x in range(0, len_bgs)], bgs)
#             bgs.append(paddings)
#         padded.append(bgs)
#     return padded    


def DTWDistance(s1, s2, w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            
    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):

    LB_sum=0
    for ind,i in enumerate(s1):
        lower_bound = upper_bound = 0
        radius = s2[(ind-r if ind-r>=0 else 0):(ind+r)]
        if len(radius)!=0:
            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return math.sqrt(LB_sum)

def k_means_clust(data,num_clust,num_iter,w=5):
  
    centroids = random.sample(data, num_clust)
       
    counter=0
    for n in range(num_iter):
        print('round ', n)
        counter+=1
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                lb = LB_Keogh(i,j,5)
                if lb<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind            
            assignments.setdefault(closest_clust,[])
            assignments[closest_clust].append(ind)
    
        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=np.zeros(len(data[0]))
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
                
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    
    return centroids,assignments

if __name__ == "__main__":
    start = time.time()
    X_pd = pickle.load(open('/Users/wang/data/OpenAPS/sgvs_train_2018-1.pkl', 'rb'))
    X = X_pd['sgvs'].values
    X = [[int(i) for i in j] for j in X ]
    print(np.array(X).shape)
    centroids, assignments = k_means_clust(X, 10, 10)
    X_pd['group'] = None
    for key,values in assignments.items():
        for v in values:
            X_pd.ix[v, 'group'] = key
    
    # print(X_pd['group'].head(1000))

    print('centroids/n', centroids)
    pickle.dump(X_pd, open('/Users/wang/data/OpenAPS/sgvs_train_2018-1-cluster-10.pkl', 'wb'))
    end = time.time()
    print('time used', end-start)
