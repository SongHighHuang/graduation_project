import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
import kd_tree_1
import time

#const var

#globe var

meanP=0
meanQ=0



#element function

def matrix_rotate_x_fun(theta):
    ans=np.array([[1,             0,            0,0],
                  [0,np.cos(theta) ,np.sin(theta),0],
                  [0,-np.sin(theta),np.cos(theta),0],
                  [0,             0,            0,1]
                  ])
    return ans

def matrix_rotate_y_fun(theta):
    ans=np.array([[np.cos(theta),0, -np.sin(theta),0],
                  [0,            1,              0,0],
                  [np.sin(theta),0,  np.cos(theta),0],
                  [0,            0,0,              1]
                  ])
    return ans

def matrix_rotate_z_fun(theta):
    ans=np.array([[ np.cos(theta),np.sin(theta),0,0],
                  [-np.sin(theta),np.cos(theta),0,0],
                  [0,             0,            1,0],
                  [0,             0,            0,1]
                  ])
    return ans

def rotate_on_axes(mrotate,data,theta):
    theta=-theta*np.pi/180
    rows=data.shape[0]
    row_ones=np.ones(rows)
    new_data=np.c_[data,row_ones.T].dot(mrotate(theta))
    return new_data[:,:3]

def rotate_on_Allaxes(data,alpha,beta,theta):
    data = rotate_on_axes(matrix_rotate_x_fun, data, alpha)
    data = rotate_on_axes(matrix_rotate_y_fun, data, beta)
    data = rotate_on_axes(matrix_rotate_z_fun, data, theta)
    return data

def T_transform_Tmatrix(t):
    tm=np.eye(4)
    for i in range(3):
        tm[3][i]=t[i]
    return tm

def translation_point_set(data,t):
    mt=T_transform_Tmatrix(t)
    rows=data.shape[0]
    row_ones=np.ones(rows)
    new_data=np.c_[data,row_ones.T].dot(mt)
    return new_data[:,:3]


def quaternion_R(q):
    R=np.array([
        [q[0]**2+q[1]**2-q[2]**2-q[3]**2,2*(q[1]*q[2]-q[2]*q[3]),2*(q[1]*q[3]+q[0]*q[2])],
        [2*(q[1]*q[2]+q[2]*q[3]),q[0]**2-q[1]**2+q[2]**2-q[3]**2,2*(q[2]*q[3]-q[0]*q[1])],
        [2*(q[1]*q[3]-q[0]*q[2]),2*(q[2]*q[3]+q[0]*q[1]),q[0]**2-q[1]**2-q[2]**2-q[3]**2]
    ])
    return R

def V_matrixQ(V):
    mQ=np.array([
        [V[0,0]+V[1,1]+V[2,2],V[1,2]-V[2,1],V[2,0]-V[0,2],V[0,1]-V[1,0]],
        [V[1,2]-V[2,1],V[0,0]-V[1,1]-V[2,2],V[0,1]+V[1,0],V[0,2]+V[2,0]],
        [V[2,0]-V[0,2],V[0,1]+V[1,0],V[1,1]-V[0,0]-V[2,2],V[1,2]+V[2,1]],
        [V[0,1]-V[1,0],V[0,2]+V[2,0],V[1,2]+V[2,1],V[2,2]-V[0,0]-V[1,1]]
             ])
    return mQ

def eulerAngle_to_quaternion(eulerAngle):
    yaw,pitch,roll=np.deg2rad(eulerAngle)
    cosRoll=np.cos(roll*0.5)
    sinRoll=np.sin(roll*0.5)

    cosyaw=np.cos(yaw*0.5)
    sinyaw=np.sin(yaw*0.5)

    cosPitch=np.cos(pitch*0.5)
    sinPitch=np.sin(pitch*0.5)

    qw = cosRoll * cosyaw * cosPitch + sinRoll * sinyaw * sinPitch
    qx = cosRoll * sinyaw * cosPitch + sinRoll * cosyaw * sinPitch
    qy = cosRoll * cosyaw * sinPitch - sinRoll * sinyaw * cosPitch
    qz = sinRoll * cosyaw * cosPitch - cosRoll * sinyaw * sinPitch

    return np.array([qw,qx,qy,qz])

def quaternion_to_eulerAngle(q):
    #roll (x-axis rotation)
    w,x,y,z=q
    roll=np.arctan2(2*(w*z+x*y),1.0-2.0*(z*z+x*x))

    # pitch (y-axis rotation)
    sinp=2.0*(w*x-y*z)
    if np.fabs(sinp)>=1:
        yaw=np.copysign(np.pi/2,sinp)
    else:
        yaw=np.arcsin(sinp)

    #yaw (z-axis rotation)
    pitch=np.arctan2(2.0*(w*y+z*x),1.0-2.0*(y*y+x*x))
    return np.degrees([yaw,pitch,roll])

#tool funtion

def loadData(filename):
    fin=open(filename,"r")
    lines=fin.readlines()
    cnt=len(lines)
    data=np.zeros([cnt,3])
    for i in range(cnt):
        data[i]=lines[i].split(' ')
    fin.close()
    return data


def outputData(filename,data):
    fout=open(filename,"w")
    for i in range(data.shape[0]):
        fout.write(str(data[i][0])+' '+str(data[i][1])+' '+str(data[i][2])+'\n')
    fout.close()

def nearest_point(P,Q):
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

def nearest_point_1(P,Q):
    dis=np.zeros(P.shape[0])
    index=np.zeros(Q.shape[0], dtype=np.int)
    kdtree=spatial.KDTree(Q)
    for i in range(P.shape[0]):
        d_error,ans=kdtree.query(P[i,:])
        dis[i]=d_error
        index[i]=ans
    return dis, index

def nearest_point_2(P,Q):
    dis = np.zeros(P.shape[0])
    pp=np.zeros((P.shape[0],P.shape[1]))
    kdtree=kd_tree_1.kd_Tree()
    kdroot=kdtree.create_KDTree(Q)
    for i in range(P.shape[0]):
        d_error,ans=kdtree.findNearestNode(kdroot,P[i,:])
        dis[i]=d_error
        pp[i]=ans
    return dis,pp

def adjust_points_set(A,B):
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

def renew_mean(P,Q):
    global meanP
    global meanQ
    meanP=np.mean(P,axis=0)
    meanQ=np.mean(Q,axis=0)

def normal_gravity(P,Q):
    P=P-meanP
    Q=Q-meanQ
    return P,Q

def normal_gravity_single(p):
    meanp=np.mean(p,axis=0)
    return p-meanp

def plot_3d_2(P,Q,theta,n):
    P=rotate_on_axes(matrix_rotate_x_fun,P,theta)
    Q=rotate_on_axes(matrix_rotate_x_fun,Q,theta)
    fig = plt.figure(n)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='.')
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='b', marker='*')
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
    ax.set_xlabel('Z Label')

def calculate_T(R):
    T=meanQ.T-np.dot(R,meanP.T)
    return T

def calculate_quaternion(P,Q):

    k1=P.shape[0]
    renew_mean(P,Q)
    data_g,data_p=normal_gravity(P,Q)
#    data_g,data_p=P,Q
    dis,index=nearest_point(data_g,data_p)
#    dis,data_pp=nearest_point_2(data_g,data_p)

    data_pp=data_p[index,:]

    V=(np.dot(data_g.T,data_pp))/k1

    matrix_Q=V_matrixQ(V)

    lambdas,D2=np.linalg.eig(matrix_Q)
    ind=np.argmax(lambdas)
    q=D2[:,ind]
    err=np.sum(dis)/dis.shape[0]

    return q,err,data_g,data_p,data_pp

def renew_target_point_set(R,data_p,data_pp):
    data_p=np.dot(data_p,R)
    data_pp=np.dot(data_pp,R)

#    data_p=translation_point_set(data_p,T)
#    data_pp=translation_point_set(data_pp,T)

#    data_p=normal_gravity_single(data_p)
#    data_pp=normal_gravity_single(data_pp)
    return data_p,data_pp

def Rescaling_data(data):
    minD=np.min(data)
    maxD=np.max(data)
    return (data-minD)/(maxD-minD)

def Rescaling_data_1(data):
    for i in range(data.shape[1]):
        minD=np.min(data[:,i])
        maxD=np.max(data[:,i])
        data[:,i]=(data[:,i]-minD)/(maxD-minD)
    return data

def MeanNormalization(data):
    meand=np.mean(data,axis=0)
    data=data-meand
    maxd=np.max(data)
    mind=np.min(data)
    diff=maxd-mind
    data=data/diff
    return data

def adjustNumber(data,controlpoint):
    m=len(data)
    if m<=controlpoint:
        return data
    sample=random.sample(range(m),controlpoint)
    data=data[sample,:]
    return data
#main funtion

def icp(src,dst,maxIteration=50,tolerance=0.001):
    cnt=0
    last_error=0
    data_p,data_g=adjust_points_set(src,dst)
    last_R=np.eye(3)

    for i in range(maxIteration):
        print("This is "+str(cnt)+" ,and error is "+str(last_error))
        cnt=cnt+1
        q,err,data_g,data_p,data_pp=calculate_quaternion(data_g,data_p)
#        print("meanP="+str(meanP)+", meanQ= "+str(meanQ)+"\n")

        R=quaternion_R(q)

        data_p,data_pp=renew_target_point_set(R,data_p,data_pp)
        last_R=np.dot(last_R,R)

        if np.abs(err-last_error)<tolerance:
            break
        last_error=err
    return last_error,last_R

if __name__=='__main__':


    data_g=np.array(loadData("face1.asc"))
#    data_g=Rescaling_data_1(data_g)
    data_g=MeanNormalization(data_g)
    data_g=adjustNumber(data_g,1200)
    data_p=rotate_on_Allaxes(data_g,30,25,-3)

#    data_p=translation_point_set(data_p,[40,20,30])
    plt.ion()
    plot_3d_2(data_g, data_p, 0, 1)

    ts=time.time()
    err,R=icp(data_p,data_g,50,0.00001)
    te=time.time()

    print(te-ts,'s')

    data_p=np.dot(data_p,R)
    plot_3d_2(data_g,data_p,0,2)



    plt.ioff()
    plt.show()






















