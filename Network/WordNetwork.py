# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:49:21 2018

@author: TrungTran
"""
import numpy as np
import pandas as pd
import matplotlib as plt
     
def BFS(G, start):
    '''duyệt cây sử dụng giải thuật BFS
    Parameters:
    -``G``: là đồ thị đầu vào
    -``start``: Là điểm xuất phát
    '''
    visited = []
    queue = []
    queue.append(start)
    while len(queue) != 0:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
        for j  in range(len(G[vertex])):
            if G[vertex][j] not in visited:
                queue.append(G[vertex][j])
                visited.append(G[vertex][j])
        
    return visited
def connectedComponent(G, numNode):
    '''duyệt BFS để tìm  các thành phần liên thông
    Parameters:
    -``G``: là đồ thị đầu vào
    '''
    visited = [0]*numNode
    start = []
    start.append(0)
    numCompts  = 0
    connetedCompt = [[]]
    while True:  
        connetedCompt.append([])
        tree =  BFS(G, start[numCompts])
        connetedCompt[numCompts].extend(tree)
        for i in range(len(tree)):
            visited[tree[i]] = 1
        numCompts += 1
        if(visited.count(0) == 0):
            break
        start.append(visited.index(0)) 
    print('So luong cac thanh phan lien thong: ' + str(numCompts))
    return connetedCompt

def mutualString(word1,word2):
    ''' dùng để tính xem hai từ có  khác nhau tại một vị trí không
    nếu có trả về true nếu không trả về false
    parameters:
    -``word1`` là từ số một
    -``word2`` là từ số hai
    '''
    count  = 0
    for i in range(5):
        if word1[i] == word2[i]:
            count  += 1
    if count == 4:
        return True
    else:
        return False
def findPath(G,dataset, numNode):
    ''' Tìm đường đi ngắn nhất của hai điểm đầu và điểm cuối đưa ra đường đi  
    dựa vào thuật toán duyệt theo chiều rộng BFS  cho đến khi gặp điểm end
    Parameters:
    -``G``: là đồ thị đầu vào
    -``start``: Là điểm xuất phát
    -``end`` : Là điểm cuối trong đường đi
    '''
    path = []
  #  start = 'there'
    start = input('Nhap tu bat dau: ')
    end = input('Nhập tu ket thuc: ' )
    start = np.argwhere(dataset  == start)[0][0]
    end = np.argwhere(dataset  == end)[0][0] 
  #  path = findPath(G,start, end ) 
    
    visited = []
    queue = []
    queue.append(start)
    preVertex = np.zeros(numNode, dtype = int)
    preVertex[start] = -1
    path = []
    path.append(end)
    flag = False
    while len(queue) != 0:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
        for j  in range(len(G[vertex])):
            if G[vertex][j] not in visited:
                preVertex[G[vertex][j]] = vertex
                queue.append(G[vertex][j])
                visited.append(vertex)
                if G[vertex][j] == end:
                    flag = True
                    while preVertex[end] != -1:
                        path.append(preVertex[end] )
                        end = preVertex[end]
    if flag == False:                        
        print('Khong co đuong di')
    print('Duong di ngan nhat:')
    print(dataset[path[::-1]])

def girvan_Newman(G, numNode):
    '''Tính betweenness cho đồ thị G sử dụng thuật toán girvan_newman
    trả về betweenness
    Parameters:
    -``G``: là đồ thị  liên thông cần tính Betweeness
    -``numNode``:  là  số lượng đỉnh trong đồ thị
    '''
    betweenness = {}
    for i in G:
        betweenness[i] = []
        for j in range(len(G[i])):
            betweenness[i].append(0) 
    for root in G:  
        # tính nhãn các cạnh trong thuật toán girman-newman
        degreeEdges= Label(G, root,numNode );
        # duyệt để tính betweeness
        for i in G:
            for j in range(len(G[i])):
                for k  in range (int(0.5*len(degreeEdges[i]))):
                    if G[i][j] == degreeEdges[i][2*k]:
                        # betwenness là bằng  một nửa tổng các nhãn cạnh
                        betweenness[i][j] +=  0.5*degreeEdges[i][2*k+1]
    return betweenness

def Label(G, start, numNode):
    ''' Thuật toán tính nhãn của các cạnh với một đỉnh xuất phát
     trả về  nhãn các cạnh
    Parameters:
    -``G``: là đồ thị  liên thông cần tính Betweeness
    -``start``:  là đỉnh xuất phát
    -``numNode``:  là  số lượng đỉnh trong đồ thị
    '''
    visited = []
    queue = []
    edges = {}
    degreeEdges = {}
    degreeNodes = {}
    level = {}
    newDegreeNodes  ={}
    # khởi tạo các giá trị ban đầu cho các biến
    for i in G:
        edges[i] = []
        degreeEdges[i] = []
        degreeNodes[i] = 1.0
        newDegreeNodes[i] = 1.0
        level[i] = []
    
    # Duyệt BFS để cho mỗi nốt để đánh dấu các node = số lượng shorttest path 
    queue.append(start)
    level[start] = 0
    while len(queue) != 0:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.append(vertex)
        for j in range(len(G[vertex])):
            if (G[vertex][j] in visited) and (level[G[vertex][j]] == level[vertex] + 1):
                degreeNodes[G[vertex][j]] += 1 # gán số lượng shorttest path
                edges[vertex].append(G[vertex][j])
                edges[G[vertex][j]].append(vertex)
            else:
                if G[vertex][j] not in visited:
                    queue.append(G[vertex][j])
                    visited.append(G[vertex][j])
                    degreeNodes[G[vertex][j]] = 1  # gán số lượng shorttest path
                    level[G[vertex][j]] = level[vertex] + 1
                    edges[vertex].append(G[vertex][j])
                    edges[G[vertex][j]].append(vertex)

    #travel bottom-up for relable edges and nodes
    visited.reverse()
    for i in range(numNode):
        curentNode = visited[i]
        for j in range (len(edges[curentNode])):
            if level[edges[curentNode][j]]>level[curentNode] :
                degreeEdges[curentNode].append( edges[curentNode][j]) #Tính label cho cạnh
                temp1 = newDegreeNodes[edges[curentNode][j]]/degreeNodes[edges[curentNode][j]]
                degreeEdges[curentNode].append(temp1) #Tính label cho cạnh
                temp  = edges[curentNode][j]
                degreeEdges[temp].append(curentNode) #Tính label cho cạnh
                degreeEdges[temp].append(temp1) #Tính label cho cạnh
                newDegreeNodes[curentNode] +=degreeEdges[curentNode][-1] # cập nhật lable cho đỉnh
                
    return degreeEdges
def main():
     #import dataset 
    dataset = np.loadtxt("sgb-words.txt", dtype = str, delimiter ='\n' );
    numNode = dataset.size
    G = {}
    for i in range(numNode): 
        G[i] = []
        for j in range(numNode):
            if mutualString(dataset[i], dataset[j]):
                G[i].append(j)
       
    # count the number of connected components
    connetedCompt = connectedComponent(G, numNode)
    numCompts = len(connetedCompt) - 1
    # nhập hai hai từ bắt đầu và hai từ xuất phát và từ xuất phát dưa đường đi ngắn nhất:
    findPath(G, dataset, numNode) 
    #tính betweenness sử dụng thuật toán girvan-newman cho đồ thị  G 
    maxBetweeness = 0.0
    indexMax = [-1, -1]
    for i in range(numCompts):
        #duyệt từng thành phần để tính betwennesss
        print('canh co betweenness lon nhat la: '+str(dataset[2344])+'-'+str(dataset[598])+ '\t betweeness: 208275.41767595825 (ket qua lan chay truoc)')
        connectedGraphs = {}
        numNode1 = len(connetedCompt[i])
        if numNode1 == 1:
            continue
        for j in connetedCompt[i]:
            connectedGraphs[j]= G[j]
        print("dang thuc hien tinh cho thanh phan lien thong thu : "+str(i+ 1) + ' so luong dinh: ' + str(numNode1))
        betweeness = girvan_Newman(connectedGraphs, numNode1)
        for j in betweeness:
            if max(betweeness[j]) > maxBetweeness:
                maxBetweeness = max(betweeness[j])
                indexMax = [j, G[j][np.argmax(betweeness[j])]]
        print('canh co betweenness lon nhat la: '+str(dataset[indexMax[0]])+'-'+str(dataset[indexMax[1]])+'\t max betweeness: '+ str(maxBetweeness))
    print('done') 

if __name__ == '__main__':      
    main()