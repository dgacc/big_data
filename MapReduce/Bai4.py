from pyspark import SparkConf, SparkContext
import sys
from collections import defaultdict

def getData(sc,filename):
    data = sc.textFile(filename)
    data = data.map(lambda l: l.split("\t"))
    data = data.map(lambda l: (l[0], l[1].split(','))) 
    return data 

def mapFriends(line):
    friends = line[1]
    for i in range(len(friends)):
        if friends[i] == '':
            return
        yield (int(line[0]) , (int(friends[i]), 0)) 
        for j in range(i+1, len(friends)):
                yield (int(friends[i]) , (int(friends[j]),1)) 
                yield(int(friends[j]), (int(friends[i]),1))

def findListFriend(line):
    friendDict = defaultdict(int) 
    directFriend = set()
    for elem in line[1]:
        if elem[1] == 0: 
            directFriend.add(elem[0])
        else:
            friendDict[elem[0]] += 1
    sortKeys = [v[0] for v in sorted(friendDict.items(), key=lambda(k,v): (-v,k))] 
    count = 0
    friendMayKnowList = []
    for key in sortKeys:
        if count >= 10:
            break
        if key not in directFriend:
            friendMayKnowList.append(key)
            count += 1
    return (line[0], friendMayKnowList)
            
if __name__ == "__main__":
    conf = SparkConf()
    sc   = SparkContext(conf=conf)
    filename = sys.argv[1]
    data = getData(sc, filename)
    mapData = data.flatMap(mapFriends).groupByKey()
    getFriends = mapData.map(findListFriend)
    output = open(sys.argv[2],'a')
    for f in getFriends.sortByKey().collect():
        output.write( "%d\t%s\n" % (f[0], ",".join([str(fr) for fr in f[1]])))
    output.close()
    sc.stop()
