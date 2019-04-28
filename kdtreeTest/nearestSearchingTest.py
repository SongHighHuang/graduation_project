import numpy as np
import kd_tree_1 as kdtree
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

def bruteForce_nearest(target,points):
    m,n=np.shape(points)
    minDis=np.inf
    index=-1
    for i in range(m):
        dis=np.linalg.norm(target-points[i],ord=2)
        if dis<minDis:
            index=i
            minDis=dis
    return index,minDis

def bruteForce_nearest_Fast(target,points):
    dis=np.linalg.norm(points-target,ord=2,axis=1)
    mindis=np.min(dis)
    index=np.argmin(dis)
    return index,mindis

def kdtree_nearest(target,points):
    kdt=kdtree.kd_Tree()
    kdr=kdt.create_KDTree(points)
    dis,index=kdt.findNearestNode(kdr,target)
    return index,dis

def kdtree_nearest_scipy(target,points):
    kd=spatial.KDTree(points)
    dis,index=kd.query(target)
    return index,dis


if __name__=='__main__':

    arr_N=np.arange(100,10000,200)
    nlen=len(arr_N)
    funs=[bruteForce_nearest,kdtree_nearest,kdtree_nearest_scipy]
    flen=len(funs)
    times=np.ones((nlen,flen))
    start=-5
    end=5

    for i in range(nlen):
        points=create3Ddata(arr_N[i],3,start,end)
        target=np.random.rand(3)*(end-start)+start
        for j in range(flen):
            timeS=time.time()
            index,dis=funs[j](target,points)
            timeE=time.time()
            timeD=timeE-timeS
            times[i,j]=timeD
    colors=['green','red','blue']
    labels=['bruteForce_nearest','kdtree_nearest','kdtree_nearest_scipy']
    markers=['o','*','.']
    linestyles=[':','-.','-']
    plt.title('searching speed')


    for i in range(flen):
        plt.plot(arr_N,times[:,i],color=colors[i],marker=markers[i],linestyle=linestyles[i],label=labels[i])

    plt.legend()
    plt.xlabel('numbers of poins')
    plt.ylabel('time')
    plt.savefig('nearest.png')
    plt.show()


    '''
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2],c='r',marker='o')
    ax.scatter(target[0],target[1],target[2],c='b',marker='*')
    ax.scatter(data[index,0],data[index,1],data[index,2],c='y',marker='^')
    plt.show()
    '''

