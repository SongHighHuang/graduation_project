import numpy as np
import KNNSearch_key as knns

ss=2

#tool funtion
def x_row_col(x):
    # 计算x的行数和列数
    if len(x.shape) > 1:
        return x.shape[0], x.shape[1]
    return 1, len(x)  # 当为只有一组数据时的输出

def knnsearch(origin,target,k):
    kdtree=knns.KDTree(target)
    m,n=x_row_col(origin)
    ans=[]
    for i in range(m):
        tmp=kdtree.search_knn_(origin[i,:],k)
        ans.append(tmp)
    return ans

def split_in_v(x):
    x=np.array(x)
    m,n,k=np.shape(x)
    x0=[]
    x1=[]
    for i in range(m):
        x0.append(x[i,:,0].astype(np.int))
        x1.append(x[i,:,1])
    return x0,x1

def lsqnormest(data,k):
    '''
    :param data: mxn 的数值参数
    :param k:  近邻参数
    :return:   法向量
    '''
    m,n=x_row_col(data)
    normalvectors=np.zeros((m,n))
    neighbors=knnsearch(data,data,k)
    for i in range(m):
        k_kpoints_dis=neighbors[i]
        k_kpoints=np.zeros(len(k_kpoints_dis),dtype=np.int)
        for j in range(len(k_kpoints_dis)):
            k_kpoints[j]=k_kpoints_dis[j][0]

        k_points=data[k_kpoints]

        p_mean=np.mean(k_points,axis=0)
        cov_mat=np.cov(np.transpose(k_points),ddof=0)
        lambdas,vectors=np.linalg.eig(cov_mat)
        idx=np.argmin(lambdas)
        normalvectors[i]=vectors[:,idx]

        flag=data[i]-p_mean
        if np.sign(np.dot(normalvectors[i],flag))<0:
            normalvectors[i]=-normalvectors[i]
    return normalvectors,neighbors

def match_kDtree_by_me(P,Q):
    '''
        KDTree 匹配 （采用自己编写的kdTree ）
        :param P: 点源 nx3
        :param Q: 目标源 nx3
        :return: dis nx1对应的距离 index nx1对应点的索引
    '''
    dis = np.zeros(P.shape[0])
    indix=np.zeros(P.shape[0],dtype=np.int)

    kdtree=knns.KDTree(Q)

    for i in range(P.shape[0]):
        tmp=kdtree.search_knn_(P[i,:],1)
        indix[i],dis[i]=tmp[0]
    return dis,indix

if __name__=='__main__':
    p=np.random.randint(1,10,(20,3))
    q=np.random.randint(1,10,(20,3))
    dis,index=match_kDtree_by_me(p,q)
    print(dis)
    print(p[index])