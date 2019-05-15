import numpy as np
import matplotlib.pyplot as plt
import scipy_Rotation as scir

def matrix_rotate_x_fun(theta):
    '''
    计算旋转矩阵3x3
    :param theta: 延x轴的欧拉角 弧度制
    :return:      3x3的旋转矩阵
    '''
    ans=np.array([[1,             0,            0],
                  [0,np.cos(theta) ,-np.sin(theta)],
                  [0,np.sin(theta),np.cos(theta)]
                  ])
    return ans

def matrix_rotate_y_fun(theta):
    '''
       计算旋转矩阵3x3
       :param theta: 延y轴的欧拉角 弧度制
       :return:      3x3的旋转矩阵
    '''
    ans=np.array([[np.cos(theta),0,  np.sin(theta)],
                  [0,            1,              0],
                  [-np.sin(theta),0, np.cos(theta)]
                  ])
    return ans

def matrix_rotate_z_fun(theta):
    '''
       计算旋转矩阵3x3
       :param theta: 延z轴的欧拉角 弧度制
       :return:      3x3的旋转矩阵
    '''
    ans=np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta),np.cos(theta) ,0],
                  [0,             0,            1],
                  ])
    return ans





def eularToQuat(eular,order='zyx',deg=True):
    '''
        :param eulerAngle: 欧拉角1x3 角度制
        :return: 四元数
        参考网址 https://www.cnblogs.com/21207-iHome/p/6894128.html
    '''
    #map
    if deg:
        radeular=np.deg2rad(eular)
    else:
        radeular=eular
    mapeular={}
    mapeular[order[0]],mapeular[order[1]],mapeular[order[2]]=radeular

    yaw, pitch, roll=mapeular['z'],mapeular['y'],mapeular['x']

    cosRoll = np.cos(roll * 0.5)
    sinRoll = np.sin(roll * 0.5)

    cosyaw = np.cos(yaw * 0.5)
    sinyaw = np.sin(yaw * 0.5)

    cosPitch = np.cos(pitch * 0.5)
    sinPitch = np.sin(pitch * 0.5)

    qw = cosRoll * cosyaw * cosPitch + sinRoll * sinyaw * sinPitch
    qx = cosRoll * sinyaw * cosPitch + sinRoll * cosyaw * sinPitch
    qy = cosRoll * cosyaw * sinPitch - sinRoll * sinyaw * cosPitch
    qz = sinRoll * cosyaw * cosPitch - cosRoll * sinyaw * sinPitch

    return np.array([qw, qx, qy, qz])

def quatToRmatrix(q):
    '''
       将4元数转化为旋转矩阵
       :param q: 四元数 q=[w,x,y,z]
       :return:  旋转矩阵 3x3
    '''
    q0, qx, qy, qz = q

    R = np.array([
        [q0 ** 2 + qx ** 2 - qy ** 2 - qz ** 2, 2 * qx * qy - 2 * q0 * qz, 2 * qx * qz + 2 * q0 * qy],
        [2 * qx * qy + 2 * q0 * qz, q0 ** 2 - qx ** 2 + qy ** 2 - qz ** 2, 2 * qy * qz - 2 * q0 * qx],
        [2 * qx * qz - 2 * q0 * qy, 2 * qy * qz + 2 * q0 * qx, q0 ** 2 - qx ** 2 - qy ** 2 + qz ** 2]
    ])
    return R

def RmatrixToEular(Rm,order='zyx',deg=True):
    sy=np.sqrt(Rm[0,0]**2+Rm[1,0]**2)
    singular=sy<1e-6
    if not singular :
        x=np.arctan2(Rm[2,1],Rm[2,2])
        y=np.arctan2(-Rm[2,0],sy)
        z=np.arctan2(Rm[1,0],Rm[0,0])
    else:
        x=np.arctan2(-Rm[1,2],Rm[1,1])
        y=np.arctan2(-Rm[2,0],sy)
        z=0
    if deg:
        eular=np.rad2deg([z,y,x])
    else:
        eular=[z,y,x]
    mapeular={}
    mapeular['z'],mapeular['y'],mapeular['x']=eular
    return [mapeular[order[0]],mapeular[order[1]],mapeular[order[2]]]

def eularToRmatrix(eular,order='zyx',deg=True):
    if deg:
        radeular=np.deg2rad(eular)
    else:
        radeular=eular
    mapeular={}
    mapeular[order[0]],mapeular[order[1]],mapeular[order[2]]=radeular

    z, y, x=mapeular['z'],mapeular['y'],mapeular['x']
    R=matrix_rotate_z_fun(z).dot(matrix_rotate_y_fun(y)).dot(matrix_rotate_x_fun(x))
    return R

def RmatrixToQuat(Rm):
    [[Qxx, Qxy, Qxz], [Qyx, Qyy, Qyz], [Qzx, Qzy, Qzz]] = Rm
    w = 0.5 * np.sqrt(1 + Qxx + Qyy + Qzz)
    x = 0.5 * np.sign(Qzy - Qyz) * np.sqrt(1 + Qxx - Qyy - Qzz)
    y = 0.5 * np.sign(Qxz - Qzx) * np.sqrt(1 - Qxx + Qyy - Qzz)
    z = 0.5 * np.sign(Qyx - Qxy) * np.sqrt(1 - Qxx - Qyy + Qzz)
    return np.array([w, x, y, z])

def quatToEular(q,order='zyx',deg=True):
    '''
        四元数转欧拉角
        :param q: 四元数
        :return:  欧拉角 1x3 角度制
        参考网址：https://www.cnblogs.com/21207-iHome/p/6894128.html
        '''
    # roll (x-axis rotation)
    w, x, y, z = q
    roll = np.arctan2(2 * (w * z + x * y), 1.0 - 2.0 * (z * z + x * x))

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * x - y * z)
    if np.fabs(sinp) >= 1:
        yaw = np.copysign(np.pi / 2, sinp)
    else:
        yaw = np.arcsin(sinp)

    # yaw (z-axis rotation)
    pitch = np.arctan2(2.0 * (w * y + z * x), 1.0 - 2.0 * (y * y + x * x))

    if deg:
        eular=np.rad2deg([yaw,pitch,roll])
    else:
        eular=[yaw,pitch,roll]
    mapeular={}
    mapeular['z'],mapeular['y'],mapeular['x']=eular
    return [mapeular[order[0]],mapeular[order[1]],mapeular[order[2]]]



if __name__=='__main__':
    eular=[12,23,32]
    print('eular:',eular)
    print('clockwise:')
    q=eularToQuat(eular)
    print('eular->quat:')
    print(q)
    rm=quatToRmatrix(q)
    print('quat->rmatrix:')
    print(rm)
    Ieular=RmatrixToEular(rm)
    print('Rm->leu:')
    print(Ieular)
    print('anticlockwise:')
    rm=eularToRmatrix(Ieular)
    print('Ieular->rm:')
    print(rm)
    print('rm->quat:')
    Iq=RmatrixToQuat(rm)
    print(Iq)
    print('quat->eular:')
    eular=quatToEular(Iq)
    print(eular)




