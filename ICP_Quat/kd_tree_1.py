import numpy as np

#KD_Tree 的节点表示
class KD_node:
    def __init__(self,data=None,sort_dim=0,left=None,right=None,parent=None):
        self.data=data #该节点的点数
        self.sort_dim=sort_dim #改节点的分割的所在的维度
        self.right=right       #右节点
        self.left=left         #左节点
        self.parent=parent     #父节点


class kd_Tree:
    def __init__(self,data=None):
        self.data=data

    def x_row_col(self,x):
        #计算x的行数和列数
        if len(x.shape)>1:
            return x.shape[0],x.shape[1]
        return 1,len(x)#当为只有一组数据时的输出

    def findSortDim(self,keypoints):
        var=np.var(keypoints,axis=0)#计算每一列的的方差
        return np.argmax(var) #计算最大方差所在的列

    def sort_row_by_dim(self,data,dim):
        arg=np.argsort(data[:,dim])#根据dim列排序
        return data[arg]

    def create_KDTree(self,keypoints,parent=None):
        #创建一颗kd_Tree 根据 keypoints
        if len(keypoints)==0: #没数据建成空节点
            return None
        m,n=self.x_row_col(keypoints)#计算行数和列数
        if m==1:                     #只有一组数据直接生成一个叶节点输出
            node=KD_node(data=keypoints,parent=parent)
            return node
        midIndex=int(m/2)            #计算中间节点
        dim=self.findSortDim(keypoints)#寻找方差最大的维度
        sortedData=self.sort_row_by_dim(keypoints,dim)#根据方差最大的维度排序
        node=KD_node(data=sortedData[midIndex])#初始一个节点 将midIndex的点装入
        node.sort_dim=dim  #赋值分割维度
        node.left=self.create_KDTree(sortedData[:midIndex],node)#生成左子树
        node.right=self.create_KDTree(sortedData[midIndex+1:],node)#生成右子树
        node.parent=parent #赋值父节点
        return node #返回根节点

    def findNearestNode(self,root,point,norm_p=2):
        #搜寻root上最接近point的点
        #nor没_p是范数的形式默认二范数
        if root==None:
            return -1,point
        p=root

        while((p.left!=None) or (p.right!=None)):#只要p不是叶节点
            sort_dim=p.sort_dim
            if (point[sort_dim]<=p.data[sort_dim]):
                if p.left==None:
                    break
                p=p.left
            else:
                if p.right==None:
                    break
                p=p.right

        min_dis=np.linalg.norm(point-p.data,ord=norm_p)#计算找到点与最近似邻叶节点的距离
        min_subscript=p.data                           #记录当前点

        q=p
        tmp=q

        #开始回溯
        while(q!=root):
            q=tmp.parent

            tmp_dis=np.linalg.norm(point-q.data,ord=norm_p)#当前节点的距离
            if tmp_dis<min_dis:  #当前节点小于最近距离时更新最小节点
                min_dis=tmp_dis
                min_subscript=q.data

            #查找点距离当前节点构成的区域分割线的垂直距离
            sortdim_dis=np.fabs(point[q.sort_dim]-q.data[q.sort_dim])
            # 若垂直距离小于距离当前结点的距离
            # 则证明以查找点为中心，以到当前结点距离为半径画圆，会与该结点构成的区域分割线相交
            if sortdim_dis<min_dis:
                #查找子树中和的距离
                if tmp==q.left:
                    tmpResult,tmppoint=self.findNearestNode(q.right,point)
                elif tmp==q.right:
                    tmpResult,tmppoint=self.findNearestNode(q.left,point)

                #如果找到的距离比最小距离好则更新
                if tmpResult>=0 and tmpResult<min_dis:
                    min_dis=tmpResult
                    min_subscript=tmppoint
            tmp=q
        return min_dis,min_subscript

