import numpy as np
import kd_tree_1
import time
from scipy import spatial


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def create3Ddata(N=10,k=3,start=-5,end=5):
    '''
    生成随机3维数据
    :param N: 个数
    :param k: 维度
    :param start: 开始
    :param end: 结束
    :return: Nx3的矩阵
    '''
    data=np.random.rand(N,k)
    return data*(end-start)+start

def match_bruteForce(P,Q):
    '''
    暴力匹配最近点 复杂度O(n^2)
    :param P: 点源 nx3
    :param Q: 目标源 nx3
    :return: dis nx1对应的距离 index nx1对应点的索引
    '''
    dis=np.zeros(P.shape[0])
    index=np.zeros(Q.shape[0], dtype=np.int)

    for i in range(P.shape[0]):
        minDis=np.inf
        for j in range(Q.shape[0]):
            tmp=np.linalg.norm(P[i]-Q[j], ord=2)
            if minDis>tmp:
                minDis=tmp
                index[i]=j
        dis[i]= minDis
    return dis, index

def match_bruteForce_Fast(P,Q):
    '''
    暴力匹配用了最小堆 复杂度O(nlog(n))
    :param P: 点源 nx3
    :param Q: 目标源 nx3
    :return: dis nx1对应的距离 index nx1对应点的索引
    '''
    dis=np.zeros(P.shape[0])
    index=np.zeros(Q.shape[0], dtype=np.int)

    for i in range(P.shape[0]):
        x=P[:,0]-Q[i,0]
        y=P[:,1]-Q[i,1]
        z=P[:,2]-Q[i,2]
        distance=x**2+y**2+z**2
        dis[i]=np.min(distance)
        index[i]=np.argmin(distance)
    return dis, index

def match_kDtree_by_scipy(P,Q):
    '''
    KDTree 匹配 （采用scipy 库的kdTree ）
    :param P: 点源 nx3
    :param Q: 目标源 nx3
    :return: dis nx1对应的距离 index nx1对应点的索引
    '''
    dis=np.zeros(P.shape[0])
    index=np.zeros(Q.shape[0], dtype=np.int)
    kdtree=spatial.KDTree(Q)
    for i in range(P.shape[0]):
        d_error,ans=kdtree.query(P[i,:])
        dis[i]=d_error
        index[i]=ans
    return dis, index

def match_kDtree_by_me(P,Q):
    '''
        KDTree 匹配 （采用自己编写的kdTree ）
        :param P: 点源 nx3
        :param Q: 目标源 nx3
        :return: dis nx1对应的距离 index nx1对应点的索引
    '''
    dis = np.zeros(P.shape[0])
    pp=np.zeros((P.shape[0],P.shape[1]))
    kdtree=kd_tree_1.kd_Tree()
    kdroot=kdtree.create_KDTree(Q)
    for i in range(P.shape[0]):
        d_error,ans=kdtree.findNearestNode(kdroot,P[i,:])
        dis[i]=d_error
        pp[i]=ans
    return dis,pp

if __name__=='__main__':

    arr_N=np.arange(100,2000,200)
    nlen=len(arr_N)
    funs=[match_bruteForce,match_kDtree_by_me,match_kDtree_by_scipy]
    flen=len(funs)
    times=np.ones((nlen,flen))
    start=-5
    end=5

    for i in range(nlen):
        points=create3Ddata(arr_N[i],3,start,end)
        target=create3Ddata(arr_N[i],3,start,end)
        for j in range(flen):
            timeS=time.time()
            index,dis=funs[j](target,points)
            timeE=time.time()
            timeD=timeE-timeS
            times[i,j]=timeD
    colors=['green','red','blue']
    labels=['match_bruteForce','match_kDtree_by_me','match_kDtree_by_scipy']
    markers=['o','*','.']
    linestyles=[':','-.','-']
    plt.title('matching speed')


    for i in range(flen):
        plt.plot(arr_N,times[:,i],color=colors[i],marker=markers[i],linestyle=linestyles[i],label=labels[i])

    plt.legend()
    plt.xlabel('numbers of poins')
    plt.ylabel('time')
    plt.show()


