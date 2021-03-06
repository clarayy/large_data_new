#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:wanglubao
# datetime:2019/9/22 14:31
# software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial.distance as dist
import os
#加载数据
def loadDataSet(fileName):
    #data = np.loadtxt(fileName,delimiter='\t')
    #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
    data= np.load(fileName)
    print('data:',data)
    print(data.shape)
    
    return data
    #data =  np.loadtxt(fileName)

#欧氏距离计算
def distEclud(x,y):
    odis = np.sqrt(np.sum((x-y)**2))
    #print("oshi dis:",odis)
    matv = np.vstack([x,y])
    jaccard = dist.pdist(matv,'jaccard')[0].tolist()
    # print("jaccard dis:",jaccard)
    up=0
    for i in range(len(x)):
        if x[i]==y[i]:
            up = up+1
    #print("up:",up)
    down = len(x)*2-up
    jac_sim = up/down
    jac_dis = 1-jac_sim
    #print("jac_sim:",jac_sim)
    #用jaccard得到的只有两类
    return odis
# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    # 获取样本数与特征值
    m,n = dataSet.shape#把数据集的行数和列数赋值给m,n
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = np.zeros((k,n))
    # 循环遍历特征值
    for i in range(k):
        index = int(np.random.uniform(0,m))
        # 计算每一列的质心,并将值赋给centroids
        centroids[i,:] = dataSet[index,:]
        # 返回质心
    return centroids


# k均值聚类
def KMeans(dataSet,k):
    m = np.shape(dataSet)[0]
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True

    # 创建质心,随机K个质心
    centroids = randCent(dataSet,k)     #k*n
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    while clusterChange:
        clusterChange = False

        #遍历所有样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有数据找到距离每个点最近的质心,
            # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distance = distEclud(centroids[j,:],dataSet[i,:])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
                clusterAssment[i,:] = minIndex,minDist
        # 遍历所有质心并更新它们的取值
        for j in range(k):
            # 通过数据过滤来获得给定簇的所有点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[j,:] = np.mean(pointsInCluster,axis=0)
    print("Congratulation,cluster complete!")
    # 返回所有的类质心与点分配结果
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    #if n != 2:
       # print("数据不是二维的")
        #return 1

    mark = ['or','ob','og','ok','^r','+r','sr','dr','<r','pr']
    if k > len(mark):
        print("k值太大了")
        return 1
    #绘制所有样本
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    #绘制质心
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i])

    plt.show()
path  = os.getcwd()+'/research/k_means/data/'
bmname = 'WS500'
dataSet = loadDataSet(path+bmname+"_p0.1_m50_ndataset2.npy")
N = 500
k = 5
centroids,clusterAssment = KMeans(dataSet,k)
np.save(path+bmname+"_p0.1_m50_k52"+str(k)+"_cen.npy",centroids)
print("centroids=",centroids)
classresult=[]
for i in range(N):
    classresult.append(int(clusterAssment[i,0]))
print(classresult)
showCluster(dataSet,k,centroids,clusterAssment)

#food500_4_k_means的结果
# centroids= [[1.00000000e+00 0.00000000e+00 4.06250000e+00 ... 1.10625000e+01
#   9.87500000e+00 5.12500000e+00]
#  [1.88636364e+01 6.59545455e+00 3.97090909e+01 ... 1.22727273e-01
#   9.54545455e-02 4.09090909e-02]
#  [8.27160494e-01 2.46913580e-01 2.04938272e+00 ... 2.90987654e+01
#   2.59259259e+01 1.52716049e+01]
#  [2.93442623e+00 4.91803279e-02 2.57103825e+01 ... 4.91803279e-02
#   4.91803279e-02 2.18579235e-02]]
#classresult = [1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 3, 3, 1, 3, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 1, 3, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 3, 2, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 3, 1, 3, 3, 3, 3, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 1, 1, 1, 2, 2, 1, 1, 3, 3, 3, 1, 3, 1, 1, 2, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 1, 0, 2, 2, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 0, 0, 0, 0, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 1, 3, 3, 1, 1, 2, 1, 3, 3, 3, 0, 0, 0, 1, 1, 3, 1, 1, 1, 1, 0, 0, 1, 3, 3, 1, 3, 1, 1, 2, 1, 2, 2, 3, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 1, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 3, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 3, 1, 3, 3, 3, 1, 1, 2, 1, 1, 2, 3, 1, 3, 1, 2, 2, 3, 3, 2, 3, 2, 1, 2, 2, 2, 2]

#food500_5_k_means的结果
# centroids= [[8.55670103e-01 2.06185567e-01 2.38144330e+00 ... 2.61237113e+01
#   2.32783505e+01 1.35979381e+01]
#  [1.79120879e+00 0.00000000e+00 2.29340659e+01 ... 4.39560440e-02
#   2.19780220e-02 3.29670330e-02]
#  [2.96142857e+01 1.88857143e+01 3.92000000e+01 ... 3.85714286e-01
#   3.00000000e-01 1.28571429e-01]
#  [1.46986301e+01 8.83561644e-01 4.10205479e+01 ... 0.00000000e+00
#   0.00000000e+00 0.00000000e+00]
#  [3.17708333e+00 9.37500000e-02 2.73020833e+01 ... 5.20833333e-02
#   7.29166667e-02 1.04166667e-02]]
# [2, 2, 2, 3, 2, 2, 2, 3, 4, 3, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 4, 4, 2, 3, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 2, 4, 4, 2, 2, 2, 0, 2, 3, 3, 3, 3, 2, 3, 2, 2, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 2, 2, 3, 3, 1, 2, 2, 1, 2, 1, 3, 4, 3, 3, 4, 1, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 3, 4, 3, 3, 3, 1, 1, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 1, 0, 0, 0, 1, 0, 4, 4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 2, 1, 3, 3, 3, 3, 3, 4, 2, 3, 3, 2, 3, 4, 3, 4, 2, 0, 4, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4, 2, 2, 1, 3, 3, 2, 3, 4, 4, 3, 3, 3, 3, 3, 0, 0, 0, 4, 4, 4, 4, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 4, 4, 2, 2, 3, 1, 4, 2, 4, 4, 1, 4, 0, 0, 1, 4, 3, 1, 3, 3, 3, 3, 1, 3, 4, 1, 3, 1, 1, 3, 1, 3, 3, 3, 3, 0, 0, 3, 3, 1, 1, 1, 3, 4, 3, 3, 0, 4, 3, 3, 1, 1, 1, 4, 3, 3, 3, 1, 0, 0, 1, 1, 1, 1, 1, 4, 0, 0, 2, 0, 0, 0, 0, 1, 3, 0, 0, 0, 4, 4, 4, 1, 3, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 3, 0, 0, 0, 0, 4, 4, 4, 0, 0, 2, 2, 2, 2, 1, 1, 4, 3, 3, 4, 4, 3, 4, 4, 4, 4, 3, 3, 2, 3, 4, 4, 1, 2, 1, 1, 1, 2, 0, 3, 4, 1, 1, 0, 0, 0, 3, 2, 1, 2, 3, 3, 2, 0, 0, 3, 4, 4, 3, 1, 3, 2, 0, 2, 0, 0, 1, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 0, 2, 4, 2, 1, 1, 1, 1, 1, 1, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 2, 4, 4, 2, 3, 0, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 1, 2, 2, 1, 1, 1, 2, 4, 4, 1, 1, 3, 3, 0, 0, 0, 0, 4, 4, 3, 2, 3, 2, 3, 1, 3, 1, 1, 4, 3, 3, 0, 2, 3, 0, 4, 3, 1, 2, 0, 0, 1, 4, 0, 1, 0, 3, 0, 0, 0, 0]

#BA1000_5_k_means:
#classresult = [1, 3, 1, 2, 3, 3, 1, 3, 3, 0, 3, 4, 3, 0, 4, 4, 4, 4, 4, 2, 3, 0, 4, 2, 0, 4, 0, 3, 4, 1, 1, 4, 3, 3, 2, 4, 3, 1, 4, 0, 1, 3, 3, 0, 0, 4, 1, 0, 3, 0, 1, 4, 1, 3, 2, 0, 0, 4, 3, 1, 1, 2, 0, 4, 4, 4, 1, 4, 0, 3, 1, 1, 3, 3, 3, 3, 4, 3, 4, 1, 0, 0, 1, 1, 4, 3, 0, 3, 4, 3, 4, 3, 1, 1, 1, 2, 3, 0, 4, 2, 1, 4, 2, 4, 4, 1, 3, 3, 1, 1, 0, 2, 4, 3, 3, 4, 2, 4, 1, 0, 1, 4, 0, 0, 3, 1, 4, 3, 2, 4, 4, 4, 0, 4, 4, 4, 0, 0, 3, 4, 1, 1, 3, 4, 1, 4, 3, 0, 1, 1, 3, 3, 4, 1, 0, 4, 3, 4, 0, 4, 0, 3, 0, 4, 3, 2, 4, 1, 1, 4, 0, 3, 2, 4, 1, 4, 4, 4, 4, 0, 3, 3, 3, 0, 1, 0, 4, 2, 2, 3, 4, 1, 2, 1, 4, 1, 4, 3, 4, 4, 2, 3, 1, 4, 0, 4, 4, 3, 3, 2, 3, 4, 3, 4, 4, 1, 4, 4, 3, 1, 0, 3, 0, 3, 3, 1, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 1, 3, 3, 1, 3, 4, 4, 4, 4, 4, 0, 4, 4, 3, 0, 4, 3, 2, 4, 2, 1, 3, 2, 1, 4, 4, 2, 1, 0, 3, 1, 1, 3, 4, 4, 4, 1, 4, 1, 3, 4, 1, 4, 1, 1, 3, 4, 4, 3, 4, 0, 3, 1, 4, 1, 3, 1, 4, 1, 4, 4, 1, 4, 3, 4, 3, 1, 4, 1, 0, 4, 4, 1, 3, 4, 4, 4, 3, 2, 3, 4, 1, 3, 2, 1, 2, 4, 0, 4, 3, 4, 1, 4, 0, 4, 3, 4, 2, 0, 0, 4, 3, 4, 1, 4, 4, 3, 2, 3, 2, 1, 2, 0, 3, 1, 3, 0, 4, 4, 1, 3, 3, 3, 4, 4, 0, 1, 4, 0, 0, 4, 3, 1, 4, 3, 3, 1, 3, 4, 4, 4, 4, 3, 3, 0, 3, 1, 3, 3, 4, 3, 4, 3, 0, 4, 3, 3, 4, 3, 2, 1, 4, 4, 0, 0, 3, 4, 0, 4, 4, 4, 3, 3, 4, 4, 2, 4, 2, 4, 4, 1, 3, 1, 1, 0, 3, 0, 4, 3, 4, 4, 4, 3, 3, 3, 2, 4, 4, 4, 4, 1, 4, 2, 0, 3, 1, 1, 1, 4, 4, 3, 1, 4, 0, 0, 4, 4, 4, 1, 3, 4, 3, 4, 3, 4, 4, 4, 4, 4, 1, 3, 4, 1, 1, 4, 3, 3, 1, 2, 4, 4, 4, 2, 1, 4, 3, 0, 4, 1, 3, 0, 0, 3, 4, 4, 0, 3, 1, 2, 1, 3, 3, 4, 4, 0, 4, 3, 4, 0, 4, 0, 1, 3, 0, 1, 0, 3, 4, 0, 4, 3, 4, 0, 4, 4, 4, 4, 4, 4, 2, 4, 3, 4, 1, 1, 4, 4, 2, 4, 1, 4, 3, 3, 4, 3, 3, 1, 2, 3, 4, 4, 1, 2, 1, 0, 3, 4, 3, 0, 3, 0, 4, 4, 3, 4, 1, 4, 1, 4, 1, 4, 4, 4, 3, 1, 1, 4, 0, 4, 3, 0, 3, 3, 3, 3, 3, 3, 4, 1, 4, 4, 1, 3, 3, 3, 2, 0, 0, 0, 3, 4, 1, 4, 4, 4, 4, 0, 3, 3, 1, 4, 4, 1, 4, 4, 0, 3, 3, 0, 1, 3, 4, 4, 3, 0, 4, 3, 1, 0, 0, 1, 4, 3, 4, 1, 4, 4, 1, 3, 4, 4, 0, 0, 4, 1, 4, 4, 3, 1, 4, 0, 0, 1, 4, 3, 3, 1, 1, 4, 4, 3, 1, 0, 0, 4, 4, 0, 4, 3, 4, 3, 3, 4, 4, 3, 4, 4, 1, 4, 0, 4, 1, 4, 4, 4, 1, 4, 1, 4, 4, 3, 3, 4, 1, 4, 3, 4, 3, 0, 0, 0, 3, 2, 1, 3, 2, 1, 0, 4, 1, 3, 3, 4, 4, 4, 2, 3, 0, 4, 0, 3, 4, 1, 0, 3, 4, 3, 2, 3, 0, 1, 4, 0, 4, 4, 4, 4, 4, 4, 3, 1, 1, 2, 3, 3, 3, 4, 0, 3, 3, 2, 4, 1, 1, 4, 3, 0, 0, 0, 0, 0, 4, 3, 3, 4, 1, 2, 4, 1, 4, 3, 4, 3, 4, 0, 3, 3, 4, 1, 2, 1, 4, 1, 3, 3, 4, 4, 1, 4, 3, 1, 4, 4, 0, 3, 4, 1, 3, 0, 1, 1, 1, 4, 0, 2, 1, 2, 0, 4, 3, 4, 3, 4, 4, 3, 3, 4, 3, 4, 0, 3, 1, 1, 1, 1, 1, 4, 3, 4, 4, 4, 1, 3, 1, 4, 4, 0, 3, 3, 3, 0, 4, 1, 3, 0, 4, 0, 4, 4, 4, 1, 4, 3, 3, 1, 4, 1, 4, 0, 3, 1, 4, 4, 4, 4, 4, 3, 4, 0, 0, 1, 1, 4, 1, 3, 0, 4, 1, 3, 1, 1, 1, 4, 3, 0, 4, 4, 3, 4, 3, 2, 3, 4, 4, 4, 1, 0, 0, 1, 1, 2, 1, 3, 2, 3, 0, 2, 4, 4, 3, 1, 4, 1, 1, 2, 1, 3, 4, 4, 4, 4, 4, 4, 1, 4, 1, 0, 3, 1, 3, 0, 3, 4, 4, 1, 0, 3, 0, 2, 3, 3, 1, 3, 1, 3, 3, 3, 1, 1, 3, 1, 4, 3, 3, 3, 1, 4, 0, 1, 3, 4, 3, 3, 1, 3, 4, 4, 0, 4, 2, 1, 0, 1, 4, 4, 4, 1, 0, 3, 4, 2, 4, 3, 3, 3, 0, 3, 3, 4, 3, 0, 0, 3, 0, 4, 4, 4, 4, 0, 0, 4, 4, 3, 1]
#BA1000_k101:
#classresult=[7, 5, 7, 8, 8, 8, 0, 0, 3, 7, 5, 9, 2, 9, 9, 9, 7, 3, 5, 9, 2, 3, 6, 2, 8, 0, 9, 5, 2, 9, 7, 6, 6, 7, 9, 5, 3, 4, 2, 8, 1, 0, 5, 5, 1, 7, 2, 5, 9, 2, 8, 9, 5, 5, 0, 0, 3, 0, 9, 2, 4, 2, 5, 2, 5, 9, 5, 3, 7, 0, 3, 7, 3, 2, 7, 9, 6, 5, 8, 3, 2, 3, 8, 2, 1, 2, 8, 9, 1, 7, 0, 2, 2, 5, 8, 7, 9, 0, 0, 6, 1, 9, 0, 2, 1, 5, 7, 5, 5, 1, 6, 8, 7, 7, 2, 4, 5, 2, 9, 7, 6, 3, 0, 3, 3, 2, 7, 0, 2, 5, 6, 8, 3, 7, 2, 9, 8, 4, 9, 7, 1, 8, 5, 0, 7, 8, 0, 7, 3, 2, 8, 4, 0, 5, 8, 3, 4, 9, 6, 5, 8, 1, 7, 1, 1, 8, 7, 1, 1, 2, 7, 0, 6, 2, 9, 4, 5, 7, 2, 6, 7, 6, 4, 6, 8, 3, 6, 6, 7, 7, 3, 5, 5, 4, 5, 7, 9, 6, 5, 5, 8, 0, 8, 8, 5, 6, 9, 2, 3, 1, 7, 9, 1, 7, 4, 3, 2, 8, 1, 6, 0, 6, 7, 8, 5, 0, 0, 9, 7, 7, 2, 5, 9, 3, 8, 8, 8, 5, 5, 4, 9, 6, 2, 4, 8, 7, 0, 7, 5, 8, 6, 7, 8, 8, 0, 1, 7, 6, 1, 2, 8, 4, 2, 7, 1, 9, 2, 9, 8, 8, 3, 3, 6, 2, 1, 2, 1, 7, 8, 8, 2, 9, 2, 7, 6, 7, 7, 4, 2, 5, 3, 2, 8, 2, 8, 8, 8, 5, 2, 2, 7, 6, 4, 7, 5, 7, 4, 7, 4, 9, 9, 0, 8, 0, 3, 6, 0, 3, 2, 8, 5, 5, 9, 1, 9, 3, 2, 9, 1, 9, 7, 4, 2, 2, 7, 9, 1, 8, 7, 9, 7, 4, 8, 1, 2, 3, 5, 3, 2, 3, 8, 3, 8, 7, 9, 9, 0, 1, 9, 1, 2, 6, 8, 7, 3, 5, 5, 0, 5, 6, 9, 3, 8, 9, 8, 1, 2, 5, 7, 6, 8, 3, 2, 7, 7, 2, 5, 9, 6, 5, 1, 4, 0, 0, 3, 8, 6, 7, 7, 3, 9, 8, 2, 4, 8, 6, 5, 5, 5, 6, 7, 0, 5, 0, 5, 9, 7, 7, 8, 0, 7, 5, 7, 2, 3, 8, 8, 7, 1, 5, 7, 4, 4, 9, 2, 3, 0, 0, 3, 6, 0, 5, 3, 7, 2, 5, 0, 5, 7, 9, 3, 8, 6, 2, 6, 7, 4, 7, 3, 9, 5, 3, 2, 9, 5, 3, 6, 4, 3, 0, 7, 2, 8, 8, 4, 6, 7, 3, 2, 7, 2, 6, 2, 7, 6, 5, 9, 8, 3, 5, 8, 0, 4, 3, 3, 0, 5, 4, 3, 9, 1, 1, 0, 5, 2, 9, 6, 2, 9, 8, 2, 7, 7, 1, 4, 7, 9, 0, 3, 6, 2, 6, 8, 7, 7, 8, 7, 9, 3, 7, 9, 6, 7, 3, 6, 6, 1, 9, 8, 6, 1, 8, 7, 9, 2, 2, 5, 6, 9, 5, 6, 7, 1, 6, 2, 7, 0, 5, 6, 6, 3, 8, 5, 7, 1, 7, 7, 6, 1, 6, 5, 9, 3, 3, 8, 8, 7, 6, 7, 9, 2, 9, 7, 7, 1, 4, 2, 5, 3, 0, 0, 8, 0, 0, 9, 2, 9, 9, 9, 4, 7, 9, 5, 8, 2, 3, 2, 6, 7, 3, 7, 5, 7, 1, 0, 0, 0, 4, 8, 7, 2, 9, 8, 8, 0, 1, 8, 2, 9, 5, 5, 6, 9, 6, 3, 2, 1, 0, 2, 5, 5, 6, 0, 2, 8, 0, 8, 8, 7, 5, 2, 3, 7, 5, 4, 9, 5, 7, 2, 3, 8, 7, 2, 3, 6, 2, 9, 2, 2, 3, 9, 5, 9, 7, 8, 8, 7, 3, 3, 6, 9, 5, 8, 5, 7, 8, 7, 8, 9, 8, 3, 2, 2, 0, 5, 5, 5, 9, 7, 7, 0, 0, 8, 7, 1, 7, 4, 5, 7, 9, 3, 9, 3, 7, 7, 3, 0, 2, 8, 3, 5, 8, 4, 9, 5, 0, 3, 3, 2, 5, 8, 6, 0, 9, 8, 7, 7, 6, 2, 5, 3, 3, 8, 2, 5, 2, 6, 4, 3, 5, 8, 2, 9, 6, 6, 9, 1, 2, 7, 1, 3, 4, 0, 2, 5, 9, 7, 6, 5, 6, 2, 5, 7, 5, 7, 7, 2, 3, 6, 7, 8, 5, 7, 6, 6, 2, 9, 5, 8, 7, 7, 5, 1, 8, 1, 6, 9, 8, 7, 9, 4, 9, 0, 3, 7, 4, 5, 2, 0, 5, 0, 6, 3, 9, 9, 5, 0, 0, 1, 9, 0, 2, 7, 6, 9, 4, 7, 3, 7, 7, 3, 1, 7, 7, 1, 7, 9, 7, 7, 6, 5, 5, 5, 9, 2, 0, 4, 8, 6, 6, 6, 9, 9, 5, 9, 8, 1, 2, 3, 9, 2, 7, 4, 6, 2, 0, 9, 7, 5, 8, 4, 9, 0, 2, 6, 0, 3, 9, 5, 9, 4, 2, 2, 6, 6, 8, 4, 6, 0, 0, 6, 8, 8, 4, 8, 5, 7, 9, 4, 6, 8, 9, 8, 2, 0, 7, 2, 5, 3, 7, 9, 2, 5, 8, 5, 3, 8, 1, 0, 0, 3, 7, 2, 7, 3, 9, 7, 5, 9, 7, 9, 9, 6, 6, 9, 3, 2, 7, 5, 3, 9, 2, 9, 7, 7, 1, 9, 2, 6, 5, 7, 2, 0, 8, 0, 9, 2, 8, 2, 8, 9, 2, 7, 9, 3, 3, 2, 3, 0, 2, 6, 8, 2, 3, 5, 8, 3, 7, 2, 3, 7, 2, 2, 8, 2, 3, 9, 3, 5, 3, 0, 2, 7, 5, 2, 5, 1, 9, 7, 7]
#BA100_k102:
#classresult = [9, 5, 9, 0, 0, 0, 7, 7, 8, 9, 5, 6, 3, 6, 1, 6, 9, 6, 1, 1, 3, 8, 8, 3, 0, 7, 6, 9, 3, 6, 9, 8, 8, 9, 6, 5, 5, 4, 3, 0, 5, 7, 5, 5, 1, 9, 3, 5, 6, 3, 0, 6, 5, 5, 7, 7, 8, 0, 6, 8, 4, 3, 5, 3, 5, 1, 5, 8, 9, 7, 8, 9, 7, 3, 9, 1, 8, 5, 0, 8, 3, 8, 0, 3, 5, 3, 0, 1, 5, 9, 7, 3, 3, 5, 0, 9, 1, 7, 7, 8, 1, 6, 7, 3, 1, 5, 9, 5, 5, 5, 8, 0, 6, 9, 1, 4, 5, 3, 6, 9, 8, 5, 7, 8, 6, 3, 9, 7, 3, 5, 8, 1, 8, 9, 3, 6, 0, 4, 6, 9, 1, 0, 5, 7, 9, 0, 7, 9, 5, 3, 0, 4, 7, 5, 6, 8, 4, 6, 8, 5, 0, 8, 9, 1, 5, 0, 9, 1, 5, 3, 9, 7, 8, 3, 1, 1, 5, 9, 3, 8, 9, 8, 4, 8, 0, 8, 8, 8, 9, 9, 8, 5, 5, 4, 5, 9, 6, 8, 5, 5, 0, 7, 0, 0, 5, 8, 1, 3, 8, 5, 9, 8, 1, 8, 4, 5, 3, 0, 1, 8, 7, 6, 9, 0, 5, 7, 7, 1, 9, 9, 1, 5, 1, 8, 0, 0, 0, 5, 5, 4, 6, 8, 3, 4, 0, 9, 7, 9, 5, 0, 8, 9, 0, 0, 7, 5, 9, 8, 1, 1, 0, 1, 3, 9, 1, 6, 3, 6, 0, 0, 8, 8, 8, 3, 5, 3, 1, 9, 0, 0, 3, 1, 3, 9, 8, 9, 9, 4, 3, 5, 8, 3, 0, 3, 0, 0, 0, 5, 3, 3, 9, 6, 4, 9, 5, 9, 4, 9, 4, 6, 6, 7, 0, 7, 1, 8, 7, 8, 3, 0, 5, 5, 6, 1, 6, 8, 1, 1, 1, 6, 9, 4, 3, 3, 9, 1, 1, 0, 9, 6, 9, 4, 0, 1, 8, 8, 5, 8, 3, 8, 0, 8, 0, 9, 6, 6, 7, 1, 1, 1, 3, 8, 0, 9, 8, 5, 5, 7, 5, 8, 6, 5, 0, 6, 6, 5, 3, 5, 9, 8, 0, 8, 3, 9, 9, 3, 5, 6, 8, 5, 5, 4, 7, 1, 5, 0, 8, 9, 9, 8, 6, 0, 3, 4, 0, 8, 5, 5, 5, 8, 9, 7, 5, 7, 5, 6, 9, 9, 0, 7, 9, 5, 9, 3, 8, 0, 0, 9, 1, 5, 9, 4, 4, 6, 3, 8, 7, 7, 8, 8, 7, 1, 5, 9, 3, 5, 7, 5, 9, 0, 8, 0, 8, 1, 8, 9, 4, 9, 8, 6, 8, 9, 3, 1, 5, 5, 8, 4, 6, 7, 9, 3, 0, 0, 4, 8, 0, 8, 3, 9, 3, 8, 1, 9, 8, 5, 1, 0, 8, 5, 0, 7, 4, 8, 8, 7, 5, 4, 8, 6, 5, 1, 7, 5, 3, 0, 8, 3, 1, 0, 3, 9, 9, 5, 4, 9, 6, 7, 8, 8, 3, 8, 0, 0, 9, 0, 9, 1, 8, 9, 6, 8, 9, 8, 8, 8, 5, 6, 0, 8, 8, 1, 9, 6, 3, 3, 5, 8, 6, 5, 8, 9, 5, 8, 3, 9, 7, 5, 8, 8, 8, 0, 5, 9, 1, 9, 9, 8, 1, 8, 5, 1, 8, 8, 0, 0, 9, 8, 9, 6, 3, 6, 9, 9, 1, 4, 3, 5, 8, 7, 7, 0, 1, 7, 6, 3, 6, 6, 6, 4, 9, 6, 5, 0, 3, 8, 1, 8, 9, 8, 9, 5, 9, 5, 7, 7, 7, 4, 0, 9, 3, 6, 0, 0, 7, 1, 0, 3, 6, 5, 5, 8, 1, 8, 8, 3, 1, 7, 3, 5, 5, 8, 7, 3, 0, 7, 0, 8, 9, 5, 3, 8, 9, 5, 4, 6, 5, 9, 3, 5, 0, 9, 3, 8, 8, 3, 6, 3, 3, 5, 6, 5, 1, 9, 0, 0, 9, 8, 8, 8, 6, 5, 0, 5, 9, 0, 9, 0, 6, 0, 8, 3, 3, 7, 5, 5, 5, 6, 9, 0, 7, 7, 0, 9, 1, 9, 4, 5, 9, 6, 8, 6, 8, 9, 9, 5, 7, 3, 0, 8, 5, 0, 4, 6, 5, 7, 8, 8, 3, 5, 0, 8, 7, 6, 0, 9, 9, 8, 8, 5, 4, 8, 0, 3, 1, 1, 8, 4, 8, 5, 0, 3, 1, 8, 8, 1, 1, 3, 9, 5, 8, 4, 7, 3, 5, 6, 9, 8, 5, 4, 3, 5, 9, 5, 9, 9, 3, 8, 8, 9, 0, 5, 9, 8, 8, 3, 6, 5, 0, 9, 9, 5, 1, 0, 1, 8, 6, 0, 9, 6, 4, 1, 7, 5, 9, 4, 5, 3, 7, 5, 7, 8, 5, 1, 6, 5, 7, 0, 1, 1, 7, 3, 9, 8, 1, 4, 9, 8, 9, 9, 8, 5, 9, 9, 5, 9, 6, 9, 9, 8, 5, 5, 5, 6, 3, 7, 4, 1, 8, 8, 8, 6, 6, 5, 6, 0, 5, 3, 8, 1, 3, 9, 4, 8, 3, 7, 1, 9, 5, 0, 4, 1, 7, 3, 8, 7, 8, 1, 5, 1, 4, 3, 3, 8, 8, 0, 4, 8, 7, 7, 8, 0, 0, 4, 0, 5, 8, 6, 4, 8, 0, 6, 0, 3, 7, 6, 3, 5, 8, 9, 6, 3, 5, 0, 5, 8, 0, 5, 7, 7, 3, 9, 3, 9, 8, 3, 9, 1, 1, 9, 1, 1, 8, 8, 7, 8, 3, 9, 5, 8, 6, 3, 6, 9, 1, 5, 6, 7, 8, 5, 9, 3, 7, 0, 7, 6, 3, 0, 3, 0, 1, 3, 9, 6, 8, 8, 3, 5, 7, 3, 8, 0, 3, 8, 1, 0, 8, 1, 3, 5, 9, 3, 3, 0, 3, 8, 6, 8, 5, 7, 7, 8, 9, 5, 3, 5, 5, 6, 9, 9]
#k102只分出来9类
#再运行一次[6, 9, 6, 4, 4, 4, 6, 3, 3, 6, 9, 1, 4, 2, 4, 2, 6, 3, 8, 2, 4, 3, 0, 4, 4, 6, 1, 6, 4, 2, 6, 0, 0, 6, 2, 9, 3, 5, 4, 4, 8, 3, 9, 9, 9, 6, 4, 9, 1, 4, 4, 2, 9, 9, 6, 3, 3, 3, 1, 4, 5, 4, 9, 4, 9, 2, 9, 3, 6, 3, 3, 6, 3, 4, 6, 2, 0, 9, 4, 3, 4, 3, 4, 5, 8, 4, 4, 2, 3, 6, 3, 4, 4, 9, 4, 6, 2, 2, 3, 0, 9, 2, 3, 3, 9, 8, 6, 9, 8, 8, 0, 4, 1, 6, 2, 5, 9, 4, 2, 6, 0, 9, 0, 3, 3, 4, 6, 4, 4, 9, 0, 4, 3, 6, 4, 1, 4, 5, 1, 6, 2, 4, 9, 6, 6, 4, 0, 6, 8, 4, 4, 5, 3, 9, 2, 3, 5, 1, 0, 9, 0, 8, 6, 2, 8, 4, 6, 8, 9, 4, 6, 3, 0, 4, 2, 8, 9, 6, 5, 0, 6, 0, 2, 0, 0, 3, 0, 0, 6, 6, 3, 9, 9, 5, 9, 6, 1, 0, 9, 9, 4, 3, 4, 4, 9, 0, 2, 4, 3, 8, 6, 0, 9, 6, 5, 8, 4, 4, 9, 0, 4, 0, 6, 4, 9, 3, 3, 2, 6, 6, 4, 9, 6, 2, 4, 4, 4, 9, 9, 5, 1, 0, 4, 2, 4, 6, 3, 6, 9, 4, 0, 6, 4, 4, 6, 9, 6, 0, 9, 2, 4, 4, 4, 6, 9, 1, 4, 2, 4, 4, 3, 3, 0, 4, 8, 4, 8, 6, 8, 4, 4, 2, 4, 6, 0, 6, 6, 5, 4, 9, 3, 4, 4, 4, 4, 4, 2, 9, 4, 4, 6, 2, 5, 6, 9, 6, 2, 6, 2, 1, 0, 4, 4, 6, 2, 0, 6, 3, 4, 5, 9, 9, 1, 2, 1, 3, 4, 2, 8, 1, 6, 5, 4, 4, 6, 2, 9, 4, 6, 1, 6, 5, 4, 9, 4, 3, 9, 3, 4, 3, 4, 3, 6, 6, 1, 1, 3, 2, 2, 2, 5, 0, 4, 6, 3, 9, 9, 4, 9, 0, 2, 8, 4, 1, 2, 8, 4, 9, 6, 0, 2, 3, 4, 6, 6, 4, 8, 1, 0, 9, 8, 5, 2, 2, 3, 4, 0, 6, 6, 3, 1, 4, 4, 5, 4, 0, 9, 8, 9, 0, 6, 6, 9, 3, 9, 1, 6, 6, 2, 6, 6, 9, 6, 4, 3, 4, 4, 6, 9, 9, 6, 5, 5, 1, 4, 3, 3, 6, 3, 0, 3, 8, 9, 6, 4, 9, 4, 9, 6, 2, 3, 4, 0, 4, 0, 6, 5, 6, 3, 2, 9, 3, 4, 2, 9, 3, 0, 5, 3, 4, 1, 4, 4, 4, 5, 0, 6, 3, 4, 6, 4, 0, 4, 6, 0, 9, 2, 4, 3, 9, 4, 3, 5, 3, 3, 0, 9, 5, 3, 1, 8, 9, 6, 9, 4, 4, 0, 4, 2, 1, 4, 6, 6, 8, 5, 6, 1, 6, 3, 0, 4, 0, 4, 6, 6, 4, 2, 4, 3, 6, 1, 0, 6, 3, 0, 0, 8, 1, 4, 4, 8, 4, 6, 2, 4, 5, 9, 0, 2, 9, 0, 6, 8, 0, 4, 6, 6, 9, 0, 0, 3, 4, 9, 6, 9, 6, 6, 0, 9, 0, 9, 4, 3, 3, 4, 4, 6, 0, 6, 2, 4, 2, 6, 6, 2, 5, 4, 9, 3, 6, 3, 4, 9, 6, 2, 4, 1, 1, 2, 5, 6, 1, 9, 4, 3, 3, 4, 0, 6, 3, 6, 9, 6, 8, 3, 3, 9, 5, 4, 6, 4, 2, 4, 5, 3, 4, 4, 4, 1, 9, 9, 0, 4, 0, 3, 5, 9, 3, 4, 9, 9, 0, 3, 4, 4, 9, 3, 4, 6, 9, 4, 3, 6, 9, 5, 2, 9, 6, 4, 3, 4, 6, 4, 3, 8, 4, 2, 4, 5, 3, 1, 9, 2, 6, 4, 4, 6, 3, 3, 0, 1, 9, 4, 9, 6, 4, 6, 4, 1, 9, 3, 4, 4, 3, 9, 9, 9, 1, 6, 6, 3, 3, 4, 6, 9, 6, 5, 9, 6, 1, 3, 1, 3, 6, 6, 3, 3, 4, 4, 3, 9, 5, 5, 2, 9, 6, 3, 3, 4, 9, 4, 0, 6, 1, 4, 6, 6, 0, 4, 9, 3, 3, 4, 4, 8, 4, 0, 5, 2, 9, 4, 4, 2, 0, 0, 2, 9, 4, 6, 8, 3, 5, 6, 4, 9, 1, 6, 0, 9, 0, 4, 9, 6, 8, 6, 6, 4, 3, 0, 6, 4, 9, 6, 0, 0, 4, 1, 9, 4, 6, 6, 9, 9, 8, 9, 0, 1, 3, 6, 1, 5, 2, 3, 3, 2, 5, 8, 4, 6, 9, 3, 0, 3, 2, 1, 9, 3, 3, 9, 2, 0, 4, 6, 0, 4, 5, 6, 2, 6, 6, 3, 8, 6, 6, 8, 6, 1, 6, 6, 0, 9, 9, 9, 2, 4, 6, 5, 4, 0, 0, 0, 2, 9, 9, 1, 4, 8, 4, 3, 2, 4, 6, 5, 0, 5, 9, 2, 6, 9, 4, 5, 2, 0, 4, 0, 3, 3, 4, 9, 0, 5, 4, 4, 0, 2, 4, 5, 0, 6, 9, 0, 4, 4, 2, 4, 9, 3, 1, 5, 0, 6, 1, 4, 4, 3, 1, 4, 9, 2, 6, 1, 5, 9, 4, 9, 3, 4, 8, 3, 3, 3, 6, 4, 6, 3, 4, 6, 8, 2, 6, 4, 4, 0, 0, 1, 3, 5, 6, 9, 3, 1, 4, 1, 6, 2, 3, 1, 4, 0, 9, 6, 4, 3, 4, 3, 2, 4, 4, 4, 4, 4, 4, 6, 1, 3, 3, 4, 3, 6, 4, 0, 4, 4, 3, 8, 4, 3, 2, 4, 3, 6, 5, 4, 4, 8, 3, 1, 3, 9, 3, 6, 4, 6, 3, 4, 9, 9, 1, 6, 6]
