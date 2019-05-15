
from icp import *





if __name__=='__main__':
    validMatchings = np.array(
        ['bruteForce', 'bruteForce_Fast', 'kDtree_by_scipy', 'kDtree', 'Delaunay', 'normalVector'])


    data_p = np.transpose(adjustNumber(np.array(loadData('bun000.asc')), 1200))
    data_g = np.transpose(adjustNumber(np.array(loadData('bun045.asc')), 1200))

    plt.ion()



    R,T,t=icp(data_g,data_p,tolerance=1e-5,matching=validMatchings[0])

    print('--------------------------------------------')
    print('ICP 计算出来的旋转矩阵R:')
    print(R)
    print('ICP计算出来的欧拉角:',scir.RmatrixToEular(R))


    # 画点云用nx3
    plot_3d_2(data_g.T, data_p.T, 0, 1)

    data_p=np.dot(R,data_p)
    data_p=data_p+T.reshape(3,1)




    plot_3d_2(data_g.T,data_p.T,0,2)#nx3
    print('time:',np.sum(t),'s')


    plt.ioff()
    plt.show()

