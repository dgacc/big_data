import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from numpy import dot
# data preprocessing
dataset = np.loadtxt("user-shows.txt", dtype = int, delimiter = " ")
shows = np.loadtxt('shows.txt', dtype = str, delimiter = '\n')
alex = np.loadtxt('alex.txt', dtype = int, delimiter = " ")
 #compute maxtrix P, Q
num_Users =  dataset[:, 1].size
num_Shows = dataset[1, :].size
P = np.zeros((num_Users, num_Users) , dtype = np.int)
Q = np.zeros((num_Shows, num_Shows) , dtype = np.int) 
for i in range(num_Users):
    count = 0
    for j in range(num_Shows):
        if dataset[i][j] > 0:
            count += 1
    P[i][i] = count        
            
for i in range(num_Shows):
    count = 0
    for j in range(num_Users):
        if dataset[j][i] > 0:
            count += 1
    Q[i][i] = count   
# compute the user-user collaborative filtering
ru_s = np.zeros(num_Shows)    
for j in range(num_Shows):
    for i in range(num_Users):
        ru_s[j] += (dot(dataset[i][:], dataset[499][:])/(np.sqrt(P[i][i])*np.sqrt(P[499][499])))*dataset[i][j]  
        
topk = np.argsort(ru_s)
topkRever = topk[::-1]
for i in range(5):
    print(shows[topkRever[i]] + '\n')
    

# Copute the item-item collaborative filtering
ru_s1 = np.zeros(num_Shows)
for j in range(num_Shows):
    for i in range(num_Shows):
        ru_s1[j] += (dot(dataset[:][i], dataset[:][j])/(np.sqrt(Q[i][i])*np.sqrt(Q[j][j])))*dataset[499][i]  
topk1 = np.argsort(ru_s1)
topkRever1 = topk1[::-1]
for i in range(5):
    print(shows[topkRever1[i]] + '\n')

#calculate   true positive
top100 = np.zeros(100, dtype = int)
top101 = np.zeros(100, dtype = int)
temp= 0
showAlex = 0
truepositive = np.zeros(19)
for i in range(num_Shows):
    if topkRever[i] < 100:    
        top100[temp] = topkRever[i]
        temp += 1 
    if  i  < 100 and alex[i ] ==  1:
        showAlex += 1
        
for i in range(1,20):
    dem = 0
    for j in  range(i):
        if alex[top100[j]] == 1:
            dem += 1
            
    truepositive[i-1] = dem/showAlex
temp = 0
for i in range(num_Shows):
    if topkRever1[i] < 100:    
        top101[temp] = topkRever1[i]
        temp += 1 
truepositive1 = np.zeros(19)

for i in range(1,20):
    dem = 0
    for j in  range(i):
        if alex[top101[j]] == 1:
            dem += 1 
    truepositive1[i-1] = dem/showAlex
#plot  true positive
#plt.subplot(211)
plt.plot(truepositive,  label="user-user")
plt.xlim((1, 20))
plt.ylabel('True Positive')
plt.xlabel('Top-k')
#plt.subplot(212)
plt.plot(truepositive1,  label="item-item")
plt.xlim((1, 20))
plt.legend()
plt.show()


 