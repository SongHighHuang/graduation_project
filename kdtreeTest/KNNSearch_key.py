from copy import deepcopy
import numpy as np
from numpy.linalg import norm

def x_row_col(x):
    # 计算x的行数和列数
    if len(x.shape) > 1:
        return x.shape[0], x.shape[1]
    return 1, len(x)  # 当为只有一组数据时的输出

def partition_sort(arr,k,key=lambda x:x):
    """
        以枢纽(位置k)为中心将数组划分为两部分, 枢纽左侧的元素不大于枢纽右侧的元素
        :param arr: 待划分数组
        :param key: 比较方式
        :return: None
    """
    start,end=0,len(arr)-1
    assert 0<=k<=end
    while True:
        i,j,pivot=start,end,deepcopy(arr[start])
        while i<j:
            #从右到左查找较小的元素
            while i<j and key(pivot)<=key(arr[j]):
                j-=1
            if i==j:break
            arr[i]=arr[j]
            i+=1
            #从左到右查找较大的元素
            while i<j and key(arr[i])<=key(pivot):
                i+=1
            if i==j: break
            arr[j]=arr[i]
            j-=1
        arr[i]=pivot

        if i==k:
            return
        elif i<k:
            start=i+1
        else:
            end=i-1

def max_heapreplace(heap,new_node,key=lambda x:x[1]):
    """
        大根堆替换堆顶元素

        :param heap: 大根堆/列表
        :param new_node: 新节点
        :return: None
    """
    heap[0]=new_node
    root,child=0,1
    end=len(heap)-1
    while child<=end:
        if child<end and key(heap[child])<key(heap[child+1]):
            child+=1
        if key(heap[child])<=key(new_node):
            break
        heap[root]=heap[child]
        root,child=child,2*child+1
    heap[root]=new_node

def max_heappush(heap,new_node,key=lambda x:x[1]):
    """
       大根堆插入元素

       :param heap: 大根堆/列表
       :param new_node: 新节点
       :return: None
    """
    heap.append(new_node)
    pos=len(heap)-1
    while 0<pos:
        parent_pos=pos-1>>1
        if key(new_node)<=key(heap[parent_pos]):
            break
        heap[pos]=heap[parent_pos]
        pos=parent_pos
    heap[pos]=new_node

class KDNode(object):
    def __init__(self,kpoint=None,label=None,left=None,right=None,axis=None,parent=None):
        """
               构造函数

               :param data: 数据
               :param label: 数据标签
               :param left: 左孩子节点
               :param right: 右孩子节点
               :param axis: 分割轴
               :param parent: 父节点
        """
        self.kpoint=kpoint
        self.label=label
        self.left=left
        self.right=right
        self.axis=axis
        self.parent=parent

class KDTree(object):
    def __init__(self,X,y=None):
        self.root=None
        self.y_valid=False if y is None else True
        self.X=X
        self.create(X,y)

    def create(self,X,y=None):
        m,n=x_row_col(X)
        index=np.arange(m)
        def create_(index,axis,parent=None):
            n_samples=np.shape(index)[0]
            if n_samples==0:
                return None
            mid=n_samples>>1
            partition_sort(index,mid,key=lambda x:X[x,axis])

            if self.y_valid:
                kd_node=KDNode(index[mid],X[index[mid]][-1],axis=axis,parent=parent)
            else:
                kd_node=KDNode(index[mid],axis=axis,parent=parent)
            next_axis=(axis+1)%k_dimensions
            kd_node.left=create_(index[:mid],next_axis,kd_node)
            kd_node.right=create_(index[mid+1:],next_axis,kd_node)
            return kd_node
        k_dimensions=np.shape(X)[1]
        if y is not None:
            X=np.hstack((np.array(X),np.array([y]).T)).tolist()
        self.root=create_(index,0)
    def search_knn_(self,point,k,dist=None):
        """
                kd树中搜索k个最近邻样本

                :param point: 样本点
                :param k: 近邻数
                :param dist: 度量方式
                :return:
        """
        def search_kun_(kd_node):
            """
                        搜索k近邻节点

                        :param kd_node: KDNode
                        :return: None
            """
            if kd_node is None:
                return
            kpoint=kd_node.kpoint
            distance=p_dist(self.X[kpoint])
            if len(heap)<k:
                max_heappush(heap,[kd_node.kpoint,distance])
            elif distance<heap[0][1]:
                max_heapreplace(heap,[kd_node.kpoint,distance])
            axis=kd_node.axis
            if abs(point[axis]-self.X[kpoint,axis])<heap[0][1] or len(heap)<k:
                search_kun_(kd_node.left)
                search_kun_(kd_node.right)
            elif point[axis]<self.X[kpoint,axis]:
                search_kun_(kd_node.left)
            else:
                search_kun_(kd_node.right)

        if self.root is None:
            raise Exception('kd-tree must be not null.')
        if k < 1:
            raise ValueError("k must be greater than 0.")

        if dist is None:
            p_dist = lambda x: norm(np.array(x) - np.array(point))
        else:
            p_dist = lambda x: dist(x, point)

        heap=[]
        search_kun_(self.root)
        return sorted(heap,key=lambda x:x[1])