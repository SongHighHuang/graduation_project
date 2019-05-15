import numpy as np
import time
import random
from numpy.matlib import repmat
import PCAnormalvectors as pcans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
import kd_tree_1
import scipy_Rotation as scir


import copy

#para
validMatchings =np.array(['bruteForce','bruteForce_Fast','kDtree_by_scipy','kDtree','Delaunay','normalVector'])


#element function
def matrix_rotate_x_fun(theta):
    '''
    计算旋转矩阵4x4
    :param theta: 延x轴的欧拉角 弧度制
    :return:      4x4的旋转矩阵
    '''
    ans=np.array([[1,             0,            0,0],
                  [0,np.cos(theta) ,np.sin(theta),0],
                  [0,-np.sin(theta),np.cos(theta),0],
                  [0,             0,            0,1]
                  ])
    return ans

def matrix_rotate_y_fun(theta):
    '''
       计算旋转矩阵4x4
       :param theta: 延y轴的欧拉角 弧度制
       :return:      4x4的旋转矩阵
    '''
    ans=np.array([[np.cos(theta),0, -np.sin(theta),0],
                  [0,            1,              0,0],
                  [np.sin(theta),0,  np.cos(theta),0],
                  [0,            0,0,              1]
                  ])
    return ans

def matrix_rotate_z_fun(theta):
    '''
       计算旋转矩阵4x4
       :param theta: 延z轴的欧拉角 弧度制
       :return:      4x4的旋转矩阵
    '''
    ans=np.array([[ np.cos(theta),np.sin(theta),0,0],
                  [-np.sin(theta),np.cos(theta),0,0],
                  [0,             0,            1,0],
                  [0,             0,            0,1]
                  ])
    return ans

def rotate_on_axes(mrotate,data,theta):
    '''
    在给定轴上旋转点云
    :param mrotate: 旋转的矩阵函数
    :param data:    数据 nxD
    :param theta:   旋转的角度 角度制
    :return:        旋转完的数据 nxD
    '''
    theta=-theta*np.pi/180 #角度转化为弧度
    rows=data.shape[0]
    row_ones=np.ones(rows)
    new_data=np.c_[data,row_ones.T].dot(mrotate(theta)) #将数据扩展为 nx(D+1)
    return new_data[:,:3]

def rotate_on_Allaxes(data,alpha,beta,theta):
    '''
    对给定点云在给定的欧拉角上旋转
    :param data:  数据nxD
    :param alpha: 延x轴旋转角度 角度制
    :param beta:  延y轴旋转角度 角度制
    :param theta: 延z轴旋转角度 角度制
    :return: nxD
    '''
    data = rotate_on_axes(matrix_rotate_x_fun, data, alpha)
    data = rotate_on_axes(matrix_rotate_y_fun, data, beta)
    data = rotate_on_axes(matrix_rotate_z_fun, data, theta)
    return data

def T_transform_Tmatrix(t):
    '''
    将平移向量1xn转化为平移矩阵4x4
    :param t:
    :return:
    '''
    tm=np.eye(4)
    for i in range(3):
        tm[3][i]=t[i]
    return tm

def translation_point_set(data,t):
    '''
    用平移向量平移点云
    :param data: 点云 nxD
    :param t: 平移向量 1x3
    :return:  平移过的点云 nxD
    '''
    mt=T_transform_Tmatrix(t)
    rows=data.shape[0]
    row_ones=np.ones(rows)
    new_data=np.c_[data,row_ones.T].dot(mt)
    return new_data[:,:3]





#tool funtion

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

def outputData(filename,data):
    '''
    输出数据
    :param filename: 文件的名称
    :param data: 传入数据 nxD
    :return: NULL
    '''
    fout=open(filename,"w")
    for i in range(data.shape[0]):
        fout.write(str(data[i][0])+' '+str(data[i][1])+' '+str(data[i][2])+'\n')
    fout.close()


#点云配准函数
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
        distance=np.sqrt(x**2+y**2+z**2)
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

def match_Delaunay(q,p,DT):
    match,mindist=match_kDtree_by_scipy(DT,np.transpose(p))
    return match,mindist

def match_normalVector_by_me(P,Q):
    normalp,_=pcans.lsqnormest(P,7)
    normalq,_=pcans.lsqnormest(Q,7)

    sump=np.sum(normalp,axis=0)
    sumq=np.sum(normalq,axis=0)
    mainNormalp=sump/np.linalg.norm(sump)
    mainNormalq=sumq/np.linalg.norm(sumq)
    cosp=np.dot(normalp,mainNormalp)
    cosq=np.dot(normalq,mainNormalq)
    p_n=len(cosp)
    index=np.zeros(p_n,dtype=np.int)
    for i in range(p_n):
        discos=np.abs(cosq-cosp[i])
        index[i]=np.argmin(discos)
    pp=Q[index]
    dis=np.linalg.norm(P-pp,axis=1)
    return dis,index

def adjust_points_set(A,B):
    '''
    调整A，B大小让他们点数相等
    :param A:
    :param B:
    :return:
    '''
    if(A.shape[0]!=B.shape[0]):
        length=min(A.shape[0],B.shape[0])
        sampleA=random.sample(range(A.shape[0]),length)
        sampleB=random.sample(range(B.shape[0]),length)
        A=np.array([A[i] for i in sampleA])
        B=np.array([B[i] for i in sampleB])
    else:
        A=np.array(A)
        B=np.array(B)
    return A,B

def normal_gravity_single(p):
    '''
    去点云的中心
    :param p: 点云 nx3
    :return: 返回结果
    '''
    meanp=np.mean(p,axis=0)
    return p-meanp

def plot_3d_2(P,Q,theta,n):
    '''
    画P点云和Q点云 在外面加plt.show() 画出
    :param P: nx3
    :param Q: nx3
    :param theta: 旋转角度
    :param n: 图片的编号
    :return: NULL
    '''
    P=rotate_on_axes(matrix_rotate_x_fun,P,theta)
    Q=rotate_on_axes(matrix_rotate_x_fun,Q,theta)
    fig = plt.figure(n)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='.')
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='b', marker='*')
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
    ax.set_xlabel('Z Label')

#归一化
def Rescaling_data(data1,data2,ax=1):
    #ax=1 3xn
    #ax=0 nx3
    minD=np.min((np.min(data1),np.min(data2)))
    maxD=np.max((np.max(data1),np.max(data2)))
    if ax==0:
        meanD=np.mean(np.vstack([data1,data2]),axis=0)
        data1 = (data1 - meanD) / (maxD - minD)
        data2 = (data2 - meanD) / (maxD - minD)
    else:
        meanD=np.mean(np.hstack([data1,data2]),axis=1)
        data1=(data1-meanD.reshape(3,1))/(maxD-minD)
        data2=(data2-meanD.reshape(3,1))/(maxD-minD)
    return data1,data2



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

def k_nearest_neighbors_by_jakob(dataMatrix,queryMatrix,k):
    '''
    :param dataMatrix:
    :param queryMatrix:
    :param k:
    :return:
    '''
    numDataPoints=len(dataMatrix)
    numQueryPoints=len(queryMatrix)

    neighborIds=np.zeros((k,numQueryPoints))
    neighborDistances=np.zeros((k,numQueryPoints))

    D=np.shape(dataMatrix)[1]

    for i in range(numQueryPoints):
        d=np.zeros(numDataPoints)
        for t in range(D):
            d=d+(dataMatrix[:,t]-queryMatrix[i,t])**2
        for j in range(k):
            s=np.min(d)
            t=np.argmin(d)
            neighborIds[j,i]=t
            neighborDistances[j,i]=np.sqrt(s)
    return neighborIds,neighborDistances

def rms_error(p1,p2):
    '''
    :param p1: nx3
    :param p2: nx3
    :return:
    '''
    dsq=np.sum(np.power(p1-p2,2),axis=1)
    ER=np.sqrt(np.mean(dsq))
    return ER

#计算旋转矩阵和平移向量

def find_optimal_transform_by_SVD(P,Q):
    '''
    用SVD法求解点云的R和T矩阵
    :param P: 3xn_P
    :param Q: 3xn_Q
    :return: R,T
    '''

    meanP=np.mean(P,axis=1)
    meanQ=np.mean(Q,axis=1)
    P_=P-meanP.reshape(3,1)
    Q_=Q-meanQ.reshape(3,1)

    W=np.dot(P_,Q_.T)
    U,S,VT=np.linalg.svd(W) #计算SVD
    R=np.dot(VT.T,U.T)
    if np.linalg.det(R)<0:
        R[2,:]*=-1
    T=meanQ.T-np.dot(R,meanP.T)
    return R,T

def find_optimal_transform_by_plane(p,q):
    '''
    :param p:        3xn
    :param q: target 3xn
    :return:
    '''
    n,_=pcans.lsqnormest(np.transpose(q),4) #nx3
    n=np.transpose(n) #3xn
    c=np.cross(p.T,n.T)#nx3
    c=np.transpose(c)#3xn
    cn=np.vstack([c,n])#6xn
    C=np.dot(cn,cn.T)#6x6
    b=-np.array([
        np.sum((p-q)*repmat(cn[0,:],3,1)*n),
        np.sum((p - q) * repmat(cn[1, :], 3, 1) * n),
        np.sum((p - q) * repmat(cn[2, :], 3, 1) * n),
        np.sum((p - q) * repmat(cn[3, :], 3, 1) * n),
        np.sum((p - q) * repmat(cn[4, :], 3, 1) * n),
        np.sum((p - q) * repmat(cn[5, :], 3, 1) * n)
    ])
    X=np.dot(np.linalg.inv(C),b)
    cx = np.cos(X[0])
    cy = np.cos(X[1])
    cz = np.cos(X[2])
    sx = np.sin(X[0])
    sy = np.sin(X[1])
    sz = np.sin(X[2])

    R = np.array([[cy * cz,cz * sx * sy - cx * sz,cx * cz * sy + sx * sz],
                  [cy * sz,cx * cz + sx * sy * sz,cx * sy * sz - cz * sx],
                  [-sy,cy * sx, cx * cy]])

    T = X[3:]
    return R,T


def DelaunayTri(data):
    tri=spatial.Delaunay(data)
    return tri.simplices

#其他函数
def calulate_Tvector(P,Q):
    meanP=np.mean(P,axis=0)
    meanQ=np.mean(Q,axis=0)
    return meanQ-meanP
def adopt_Transform(P,Q,ax=1):
    meanP=np.mean(P,axis=ax)
    meanQ=np.mean(Q,axis=ax)
    diff=meanQ-meanP
    if ax==1:
        P=P+diff.reshape(3,1)
    else:
        P=P+diff
    return P,Q

def icp(q,p,iter=200,matching='bruteForce_Fast',method=0,tolerance=5e-6,returnAll=False,display=True):
    '''

    :param q: point cloud 3xn_Q
    :param p: point cloud 3xn_P
    :param varargin:
    :return: TR  rotation
             TT  translation
             ER  error
             t   time
    [TR, TT] = icp(q,p)   returns the rotation matrix TR and translation
    '''
    # para
    '''
    validMatchings = np.array(
        ['bruteForce', 'bruteForce_Fast', 'kDtree_by_scipy', 'kDtree', 'Delaunay', 'normalVector'])
    validMinimizes = np.array(['svd', 'Quaternion'])
    '''
    # t记录时间.
    t = np.zeros(iter + 1)

    m_p,n_p=np.shape(p)
    #可跌代的数据
    pt=copy.deepcopy(p)

    #记录误差
    ER=np.zeros(iter+1)

    #初始化暂时平移向量和旋转矩阵
    last_T=np.zeros(3)
    last_R=np.eye(3)
    last_error=0

    #迭代次数
    lastiter=1

    #初始化总的平移向量和旋转矩阵
    TT=np.zeros((3,1,iter+1))
    TR=[]


    for k in range(iter):
        # 设置开始时间
        time_start = time.time()

        #最近点匹配
        #匹配是数据从3xn转置为nx3
        if matching==validMatchings[0]:
            dis,index=match_bruteForce(pt.T,q.T)
        elif matching==validMatchings[1]:
            dis,index=match_bruteForce_Fast(pt.T,q.T)
        elif matching==validMatchings[2]:
            dis,index=match_kDtree_by_scipy(pt.T,q.T)
        elif matching==validMatchings[3]:
            dis,pp=match_kDtree_by_me(pt.T,q.T)
        elif matching==validMatchings[4]:
            DT=DelaunayTri(q.T)
            dis,index=match_Delaunay(q.T,pt.T,DT)
        elif matching==validMatchings[5]:
            dis,index=match_normalVector_by_me(pt.T,q.T)
        else:
            dis,index=match_kDtree_by_scipy(pt.T,q.T)

        #计算匹配点云pp 3xn
        if matching!=validMatchings[3]:
            pp=q[:,index]
        else:
            pp=pp.T


        #计算旋转矩阵和平移矩阵
        #用3xn来计算
        if method==0:
            R,T=find_optimal_transform_by_SVD(pt,pp)
        elif method==1:
            R,T=find_optimal_transform_by_plane(pt,pp)
        else:
            R, T = find_optimal_transform_by_SVD(pt, pp)
        #更新数据
        pt=np.dot(R,pt)
        pt=pt+T.reshape(3,1)

        TT[:,:,k]=T.reshape(3,1)
        TR.append(R)
        last_R=np.dot(R,last_R)
        last_T=last_T+T

        #计算误差
        err=np.mean(dis)
        ER[k]=err

        #计算本次时间
        end_time=time.time()
        t[k]=end_time-time_start

        #显示本次循环结果和时间
        if display:
            print('循环次数：',k,' 误差：',np.abs(err-last_error),' 花费时间：',t[k],'s',' eular= ',scir.RmatrixToEular(R))


        #判断是否小于误差小于跳出
        if np.abs(err-last_error)<tolerance:
            lastiter=k+1
            break
        last_error=err

    if returnAll:
        return last_R,last_T,t,last_error,TR,TT,ER,lastiter
    return last_R,last_T,t

if __name__=='__main__':
    validMatchings = np.array(
        ['bruteForce', 'bruteForce_Fast', 'kDtree_by_scipy', 'kDtree', 'Delaunay', 'normalVector'])


    eular=[-7,8,6]
    Ttrans=[1,1,2]
    data_p=np.transpose(np.array(loadData("face1.asc"))) #3xn
    data_p=np.transpose(adjustNumber(data_p.T,1000)) #3xn
    data_g=np.dot(scir.eularToRmatrix(eular),data_p) #3xn
    data_g=data_g+np.array(Ttrans).reshape(3,1)


    data_p,data_g=Rescaling_data(data_p,data_g)#将点云的值缩小到[-1,1]
    '''
    data_p = np.transpose(adjustNumber(np.array(loadData('bun000.asc')), 1200))
    data_g = np.transpose(adjustNumber(np.array(loadData('bun045.asc')), 1200))
    '''
    plt.ion()



    R,T,t=icp(data_g,data_p,tolerance=1e-5,matching=validMatchings[5],method=0)
    '''
    print('最初的的欧拉角eular:',eular)
    print('最初的旋转矩阵R:')
    print(scir.eularToRmatrix(eular))
    '''
    print('--------------------------------------------')
    print('ICP 计算出来的旋转矩阵R:')
    print(R)
    print('ICP计算出来的欧拉角:',scir.RmatrixToEular(R))


    # 画点云用nx3
    plot_3d_2(data_g.T, data_p.T, 0, 1)

    data_p=np.dot(R,data_p)
    data_p=data_p+T.reshape(3,1)

    #data_p,data_g=adopt_Transform(data_p,data_g)


    plot_3d_2(data_g.T,data_p.T,0,2)#nx3
    print('time:',np.sum(t),'s')


    plt.ioff()
    plt.show()





































