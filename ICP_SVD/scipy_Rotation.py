from scipy.spatial.transform import Rotation
import numpy as np

'''
             eular
            /     \
           /       \
          /         \
         /           \
        /             \
       /               \
      /                 \
     /                   \
    /                     \
Rmatrix-----------------Quatition
参考：https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation
'''

def eularToQuat(eular,order='zyx',deg=True):
    r=Rotation.from_euler(order,eular,degrees=deg)
    return r.as_quat()

def quatToRmatrix(q):
    r=Rotation.from_quat(q)
    return r.as_dcm()

def RmatrixToEular(Rm,order='zyx',deg=True):
    r=Rotation.from_dcm(Rm)
    return r.as_euler(order,degrees=deg)

def eularToRmatrix(eular,order='zyx',deg=True):
    r=Rotation.from_euler(order,eular,degrees=deg)
    return r.as_dcm()

def RmatrixToQuat(Rm):
    r=Rotation.from_dcm(Rm)
    return r.as_quat()

def quatToEular(q,order='zyx',deg=True):
    r=Rotation.from_quat(q)
    return r.as_euler(order,degrees=deg)

if __name__=='__main__':
    print('Scipy toolkit')
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