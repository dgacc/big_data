import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from operator import itemgetter
from sklearn.preprocessing import normalize
from collections import Counter

# Reading dataset 
G= nx.read_edgelist('Facebook_Dataset.txt', create_using = nx.Graph(), nodetype = int)
# visualizing the data se

def friends(G, user):
    return set(G.neighbors(user))
# getting the friends of friends
def listNum(G):
    listNums = []
    node  = []
    pos = []
    for i in G:
        listNums.append(len(set(G.neighbors(i))))
        node.append(i)
    listSorted = list(np.argsort(listNums))
    for i in listSorted:
        pos.append(node[i])
    pos = pos[::-1]
    return pos
def numFreind(G):
    listNums = []
    sorList  = []
    for i in G:
        listNums.append(len(set(G.neighbors(i))))
    sorList = np.sort(listNums)
    return  Counter(sorList)
def friendsOfFriends(G, user):
    fOF = []
    for i in G.neighbors(user):
        for j in G.neighbors(i):
            if(j != i):
                fOF.append(j)
    return set(fOF)
# geting common friends
def mutual_friends(G, user1, user2):
    friends1 = set(G.neighbors(user1))
    friends2 = set(G.neighbors(user2))
    return friends1.intersection(friends2)
# getting the number of mutual friends   
def numberOfMutualFriends(G, user):
    MutualFriends = {};
    for i in G.node():
        if(i != user):
            if(i not in G.neighbors(user)):
                MutualFriends[i] = len(mutual_friends(G, user, i ))
    return MutualFriends
# getting union of set friends
def unionFriends(G, user1, user2):
    friends1 = set(G.neighbors(user1))
    friends2 = set(G.neighbors(user2))
    return friends1.union(friends2) 
# clustering coefficient
def D_c(G, Cluster):
    sG = G.subgraph(Cluster)
    if len(Cluster) > 1:
        return len(sG.edges(Cluster)) /(len(Cluster)*(len(Cluster) - 1)/ 2)
    else:
        return 0.0
def numCommonFriend(G, user):
    new_dict = dict()
    for each in G.nodes():
        if(each!=user):
            if(each not in G.neighbors(user)):
                new_dict[each] = len(mutual_friends(G,each,user))
    return new_dict


def mapSort(map):
    map = sorted(map.items(), key = itemgetter(1), reverse=True)
    return map


def recommendCommonfriends(G, user):
  
    diction = dict()
    diction = numCommonFriend(G,user)
    diction = mapSort(diction)
    recommendations = []
    for i in range(0,10):
        recommendations.append(diction[i])
    return recommendations
def optimizationFunction(index, w):
    return index[0]*w[0] + index[1]*w[1] + index[2]*w[2]
# Calculate indexes 
def calculateIndex(G, user, candidate):
    index1 = len(mutual_friends(G, user,candidate))
    index2 = D_c(G, mutual_friends(G, user,candidate))
    index3 = D_c(G, unionFriends(G, user,candidate))
    return index1,index2,index3
    
def decoding(chromosome):
    w1 = chromosome[0:8]
    w2 = chromosome[8:16]
    w3 = chromosome[16:24]
    str1 = ''
    str2 = ''
    str3 = ''
    for i in range(8):
        str1 +=str(w1[i]) 
        str2 +=str(w2[i]) 
        str3 +=str(w3[i]) 
    return [int(str1, 2),int(str2, 2), int(str3, 2) ]
def evaluation(user,friend ,candidate, population, popsize, index):
    popFiness = np.zeros(popsize, dtype = float)
    for i in range(popsize):
        w = decoding(population[i])
        popFiness[i]= fitnessFunction(user,friend ,candidate,w, index)
    return popFiness
def sortIndividual(popFitness):
    return np.argsort(popFitness)

def crossover(par1 , par2, random, crossRate, muRate):
    point = random.randint(0,23)
    offspring1 = np.ones(24, dtype = int)
    offspring2 = np.ones(24, dtype = int)
    r = random.random()
    if(r < crossRate):
        offspring1[:point]  = par1[:point]
        offspring1[point: ] = par2[point:]
        offspring2[:point ] = par2[:point]
        offspring2[point:] = par1[point:]
        if(r < muRate):
            point1 = random.randint(0,23)
            while(point1 == 7):
               point1 = random.randint(0,23)
                
            if(offspring1[point1] == 0):
                offspring1[point1] = 1
            else:
                offspring1[point1] = 0
            if(offspring2[point1] == 0):
                offspring1[point1] = 1
            else:
                offspring1[point1] = 0
        #print(offspring1)
        return [offspring1, offspring2]
       
    
    else:
        return [par1, par2]
def fitnessFunction(user,friend ,candidate, weight, index):
    Posisiton = []
    weightAverage = []
    friend1 = list(candidate)
    a =  weight / np.linalg.norm(weight)
    for i in friend1:
        weightAverage.append( optimizationFunction(index[i],a ))  
    sortedWeight = list(np.argsort(weightAverage))
    sortedWeight = sortedWeight[::-1]
    for  i in friend:
            vitri = friend1.index(i)
            Posisiton.append(sortedWeight.index(vitri))
    return np.average(Posisiton)



def GA(user,friend ,candidate, index):
  # declare parameter
  popSize = 200;
  population = []
  popFitness = np.zeros(popSize, dtype = float)
  #initilization population
  for j in range(popSize):
      chromosome = np.zeros(24, dtype = int)
      for i in range(24):
          if( random.random() < 0.5):
              chromosome[i] = 0
          else:
              chromosome[i] = 1
      chromosome[7] = 1 
      chromosome[15] = 1
      chromosome[23] = 1 
      population.append(chromosome)
  popFitness = evaluation(user,friend ,candidate, population, popSize, index)
  # sort population ansdencing order
  temp = sortIndividual(popFitness)
  
  bestIndividual = []
  bestFitness = popFitness[temp[0]]
  terminate = 0
  genaration = 0
  # evolutionary processing
  while(terminate < 5):
      genaration += 1
      offspringPop = []
      offspringFitness = np.zeros(popSize, dtype = float)
      # crossover and mutation
      for i in range(100):
          a = random.randint(0, popSize - 1)
          b = random.randint(0, popSize - 1)
          while(a ==b):
              b = random.randint(0, popSize -1)
          offsprings = crossover(population[a], population[b], random, 0.9, 0.1)
          offspringPop.extend(offsprings)
      # concate the population to offspring population
      offspringFitness =  evaluation(user,friend ,candidate, offspringPop, popSize, index)
      offspringPop.extend(population)
      conFitness = np.append(offspringFitness, popFitness)
      sortedInd = sortIndividual(conFitness)
      count  = 0
      for i in sortedInd[:popSize]:
        population[count] = offspringPop[i]
        popFitness[count] = conFitness[i]
        count += 1  
              
      if( popFitness[0] < bestFitness):  
          bestIndividual = population[0]
          bestFitness = popFitness[0]  
          terminate = 0
      else:
          terminate += 1
      print("lan lap " +str(genaration) +" tot nhat: "+str(bestFitness)) 
  return bestIndividual

def recommendation(index, candidate, weight):
    costList = []
    recommentList = []
    friend1 = list(candidate)
    for i in friend1:
        cost = optimizationFunction(index[i], weight)
        costList.append(cost)
    sortedRecom = list(np.argsort(costList))
    sortedRecom = sortedRecom[::-1]
    
    for i in candidate:
        pos = friend1.index(i)
        recommentList.append(sortedRecom.index(pos))
  #  print(recommentList[:])      
    return recommentList[:10]
def main():
    # Reading dataset 
    G= nx.read_edgelist('Facebook_Dataset.txt', create_using = nx.Graph(), nodetype = int)
    # chon user:
    user = 1000
    '''
    removed = [1,340,225,317,147,137,67,83,202,184]
    for i in  removed:
        G.remove_edge(user, i)
    G.number_of_edges()
    '''
    A = numFreind(G)
    X_x = []
    Y_y = []
    for i in A:
        X_x.append(i)
        Y_y.append(A[i])
    plt.plot(X_x, Y_y, linewidth=0.5, color='black')
    plt.savefig('occurance.pdf')
    plt.show()
    reCandidate  = friendsOfFriends(G, user)  # recommend list
    # List of freinds
    friend = friends(G, user)
    # candidate for learning weight
    candidate = reCandidate.union(friend)
    print("so ban: "+str(len(friend))+" so ung cu vien: "+str(len(candidate)))
    # calculate index for friends
    index = {}
    for i in candidate:
        index[i] = calculateIndex(G, user, i)
    # calculate index for friends
    trung = GA(user,friend ,candidate, index)
    print(decoding(trung))
    a = recommendation(index,candidate,trung)
    print(a)
if __name__ == '__main__':
    main()
 
    