import numpy as np
import copy
import time

def parttion(data, left, right):
    priov = data[left][1]
    tmp=copy.deepcopy(data[left])
    low = left
    high = right
    while low < high:
        while (low < high) and (data[high,1] >= priov):
            high -= 1
        data[low] = data[high]
        while (low < high) and (data[low,1] <= priov):
            low += 1
        data[high] = data[low]
    data[low] = tmp
    return low
def quicksort(v, left, right):
    if left < right:
        p = parttion(v, left, right)
        quicksort(v, left, p-1)
        quicksort(v, p+1, right)

def partition_knearestSearch(P,target,k):
    #P mx3
    m,n=np.shape(P)
    if n>m:
        P=np.transpose(P)
        m=n
    k=k-1
    dismap =np.array([[i, np.linalg.norm(P[i] - target, ord=2)] for i in range(m)])
    i=parttion(dismap,0,m-1)
    while i<m:
        if i==k:
            break
        elif i<k:
            i=parttion(dismap,i+1,m-1)
        else:
            i=parttion(dismap,0,i-1)
    #print(dismap)
    return dismap[:k+1,0].astype(np.int)


def partition_kMatching(P,Q,k):
    m, n = np.shape(P)
    if n > m:
        P = np.transpose(P)
        Q = np.transpose(Q)
        m = n
    ans=[]
    for i in range(m):
        tmp=partition_knearestSearch(Q,P[i],k)
        ans.append(tmp)
    return np.array(ans,dtype=int)


if __name__ == '__main__':
    p = np.random.randint(1, 10, (1000, 3))
    q = np.random.randint(1, 10, (1000, 3))
    ts = time.time()
    index = partition_kMatching(p, q, 4)
    te = time.time()
    print(te - ts, ' s')
    print(p[index[2], :])
