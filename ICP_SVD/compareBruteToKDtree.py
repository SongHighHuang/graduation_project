from icp import *

if __name__=='__main__':
    arr_N = np.arange(100, 2000, 200)
    nlen=len(arr_N)
    funs=['bruteForce','kDtree']
    flen=len(funs)
    times=np.ones((nlen,flen))

    for i in range(nlen):
        data_p = np.transpose(adjustNumber(np.array(loadData('bun000.asc')), arr_N[i]))
        data_g = np.transpose(adjustNumber(np.array(loadData('bun045.asc')), arr_N[i]))
        for j in range(flen):
            R,T,t=icp(data_g,data_p,tolerance=1e-5,matching=funs[j],display=False)
            print('success:',arr_N[i],' funs= ',funs[j])
            times[i,j]=np.sum(t)
    print(times)
    colors = ['green', 'red', 'blue']
    markers = ['o', '*', '.']
    linestyles = [':', '-.', '-']
    plt.title('ICP matching speed compare')

    for i in range(flen):
        plt.plot(arr_N, times[:, i], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=funs[i])

    plt.legend()
    plt.xlabel('numbers of poins')
    plt.ylabel('time')
    plt.show()