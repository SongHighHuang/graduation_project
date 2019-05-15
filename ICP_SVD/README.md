运行工程的方法是需要安装的python库为
pip install numpy matplotlib mpl_tookits scipy

代码：
        KNNSearch_key.py  这是k个最近点kdTree搜索类
        PCAnormalvectors.py pca列主元求点云法向量类
        iRotation.py 欧拉角 旋转矩阵 四元数 相互转化的六个函数；运行工程 python iRotation.py 
        scipy_Rotation.py scipy库编写的 欧拉角 旋转矩阵 四元数 相互转化的六个函数 python scipy_Rotation.py
        icp.py 是icp主算法代码包括SVD 和 plane匹配
        kd_tree_1.py kdTree 最近匹配所在的类
        bruteICPTest.py  暴力匹配ICP实验函数 运行 python bruteICPTest.py
        kdtreeICPTest.py  kdtree ICP 实验代码 运行 python  kdtreeICPTest.py
        
        
        
