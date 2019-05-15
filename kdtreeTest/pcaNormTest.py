import numpy as np
import time
import PCAnormalvectors as pcan
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def loadData(filename):
    '''
    载入数据
    :param filename: 文件的名称
    :return: 数据的矩阵 nxD
    '''
    fin=open(filename,"r")
    lines=fin.readlines()
    cnt=len(lines)
    data=np.zeros([cnt,3])
    for i in range(cnt):
        data[i]=lines[i].strip().split(' ')
    fin.close()
    return data

def adjustNumber(data,controlpoint):
    '''
    随机去掉点调整点的个数
    :param data: 点云 nx3
    :param controlpoint: 需要调整的个数
    :return: 调整过的个数
    '''
    m=len(data)
    if m<=controlpoint:
        return data
    sample=random.sample(range(m),controlpoint)
    data=data[sample,:]
    return data

if __name__=='__main__':
    filename='face1.asc'
    data=loadData(filename)
    data=adjustNumber(data,2000)
    m, n = np.shape(data)

    ts=time.time()
    normalvectors,_=pcan.lsqnormest(data,5)
    te=time.time()
    print('time: ',te-ts)
    terminapoints=data+normalvectors*5

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    plt.title('normal test')
    ax.scatter(data[:,0],data[:,1],data[:,2],c='b',marker='.')

    for i in range(m):
        x=[data[i,0],terminapoints[i,0]]
        y = [data[i, 1], terminapoints[i, 1]]
        z = [data[i, 2], terminapoints[i, 2]]
        ax.plot(x, y, z, c='r')
    #plt.savefig('normaltest.png')'''
    plt.show()

