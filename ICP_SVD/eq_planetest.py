from icp import *

if __name__=='__main__':
    switching=True

    if switching:
        eular = [-7, 14, 6]
        Ttrans = [1, 1, 2]
        data_p = np.transpose(np.array(loadData("face1.asc")))  # 3xn
        data_p = np.transpose(adjustNumber(data_p.T, 1000))  # 3xn
        data_g = np.dot(scir.eularToRmatrix(eular), data_p)  # 3xn
        data_g = data_g + np.array(Ttrans).reshape(3, 1)

        data_p, data_g = Rescaling_data(data_p, data_g)  # 将点云的值缩小到[-1,1]
    else:
        data_p = np.transpose(adjustNumber(np.array(loadData('bun000.asc')), 1200))
        data_g = np.transpose(adjustNumber(np.array(loadData('bun045.asc')), 1200))

    last_R1,last_T1,t1,last_error1,TR1,TT1,ER1,it1=icp(data_g,data_p,tolerance=1e-5,matching='kDtree',display=True,returnAll=True)
    last_R2, last_T2, t2, last_error2, TR2, TT2, ER2,it2 = icp(data_g, data_p, tolerance=1e-5, matching='kDtree',
                                                        display=True, returnAll=True,method=1)
    colors = ['green', 'red', 'blue']
    markers = ['o', '*', '.']
    linestyles = [':', '-.', '-']
    funs=['SVD','plane']
    plt.title('ICP matching speed compare')



    plt.plot(np.arange(0,it1),ER1[:it1], color=colors[0], marker=markers[0], linestyle=linestyles[0], label=funs[0])
    plt.plot(np.arange(0,it2),ER2[:it2], color=colors[1], marker=markers[1], linestyle=linestyles[1], label=funs[1])

    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('error')
    plt.show()