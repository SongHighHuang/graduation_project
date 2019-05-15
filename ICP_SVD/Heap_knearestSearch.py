import numpy as np
import heapq
import time
from scipy import spatial


def knearestSearch(P,target,k):
    #P mx3
    m,n=np.shape(P)
    if n>m:
        P=np.transpose(P)
        m=n

    dismap=[[i,np.linalg.norm(P[i]-target,ord=2)] for i in range(m)]
    ans=heapq.nlargest(k,dismap,key=lambda x:-x[1])
    ans=np.array(ans)
    #print(ans)
    ans=ans[:,0].astype(np.int)
    return ans

def kMatching(P,Q,k):
    m, n = np.shape(P)
    if n > m:
        P = np.transpose(P)
        Q = np.transpose(Q)
        m = n
    ans=[]
    for i in range(m):
        tmp=knearestSearch(Q,P[i],k)
        ans.append(tmp)
    return np.array(ans,dtype=int)

def lsqnormest(data,k):
    '''
    :param data: mxn 的数值参数
    :param k:  近邻参数
    :return:   法向量
    '''
    m,n=np.shape(data)
    normalvectors=np.zeros((m,n))
    neighbors=kMatching(data,data,k)
    for i in range(m):

        k_points=data[neighbors[i]]

        p_mean=np.mean(k_points,axis=0)
        cov_mat=np.cov(np.transpose(k_points),ddof=0)
        lambdas,vectors=np.linalg.eig(cov_mat)
        idx=np.argmin(lambdas)
        normalvectors[i]=vectors[:,idx]

        flag=data[i]-p_mean
        if np.sign(np.dot(normalvectors[i],flag))<0:
            normalvectors[i]=-normalvectors[i]
    return normalvectors

if __name__=='__main__':

    p=np.random.randint(1,10,(1000,3))
    q=np.random.randint(1,10,(1000,3))
    ts=time.time()
    index=kMatching(p,q,4)
    te=time.time()
    print(te-ts,' s')
    print(p[index[2],:])

    ts=time.time()
    norl=lsqnormest(p,6)
    te=time.time()
    print(norl)
    print(te-ts,'s')


    
 




