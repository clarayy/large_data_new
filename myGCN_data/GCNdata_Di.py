#节点状态做为node_labels
#单张图propagation1_6.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx
import os
import random
from itertools import islice
import torch
from scipy.sparse.coo import coo_matrix
import argparse
import math
from random import choice
import community as community_louvain
#变量：B，sn范围,每个sn张数 ，Infectionrate, Roundtime，data文件名

N = 200#记得必须改
fname = "BA200"
B = np.load(fname+".npy")#读取固定的图
Gnw = nx.DiGraph(B)
G = nx.read_edgelist('./'+fname+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())

#partition = community_louvain.best_partition(G)
#np.save("BA500_partition.npy",partition)
#partition=np.load(fname+'_k5p1'+'.npy',allow_pickle = True).item()##
graph_labels = []
graph_labels_class=[]
countedge = []
countadj = []
part=0.3
InfectionRate = 0.5#概率太大，10轮感染1400个节点
Roundtime = 3
def genGraph(sn,datadir,bmname):

#感染过程
    node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
    S = node
    I = []

    j=0
    while j<1:
        #start_node = random.choice(node)#1个初始感染节点
        start_node = sn#1个初始感染节点
        I.append(start_node)
        S.remove(start_node)
        j=j+1
    
    print(start_node)
    
    new_G_small = nx.DiGraph()
    count = [1]
    statechange = []
    edgechange = []
    edgeweight = []
    weight_s = 1
    #for i in range(Roundtime):       #####从I开始遍历，小于边的概率就感染
    while len(I)<=part*len(G.nodes()):
        for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
            if int(nbr) in I:            
                for key in datadict:
                    if int(key) in S:
                        rate = G.get_edge_data(int(nbr),int(key))['weight']   #I->S 
                        if random.random() <= rate: 
                            statechange.append(int(key))
                            #new_G_small.add_edge(int(nbr),int(key))     #I->S
        statechange=list(set(statechange))
        for i in statechange:
            S.remove(i)
            I.append(i)
        if len(I)==1:
            break
        count.append(len(I))
        statechange = []
    resultall={}
    for i in range(N):
        if i in I:
            resultall[i]=1    #I状态
        else:
            resultall[i]=0    #S状态
    new_G_small = G.subgraph(I)
    perfix = os.path.join(datadir,bmname)
    filename_node_labels = perfix + '_node_labels.txt'
    # filename_center = perfix + '_jordancenter.txt'
    # filename_center1 = perfix + '_jordancenter1.txt'
    #new_G_small表示感染图
    #new_G表示感染图加上未感染的节点，
    #将new_G中的邻接矩阵扩展到G
    new_G = nx.Graph()
    new_G.add_nodes_from(i for i in range(N))
    new_G.add_nodes_from(new_G_small.nodes())    #不增加单独节点看实验效果如何，max_nodes=100时，max graph size 是否为100
    new_G.add_edges_from(new_G_small.edges())
    #Jordan_center  = nx.center(new_G)#非全连接图不能计算

    if not nx.is_empty(new_G):
        adj_matrix = nx.adjacency_matrix(Gnw).todense()
        #Jordan_center  = nx.center(new_G_small)

        # #dynamic ages
        # frozen_graph = nx.freeze(new_G_small)#G被冷冻为frozen_graph，不会改变
        # unfrozen_graph = nx.Graph(frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
        # AS = nx.adjacency_spectrum(frozen_graph)#邻接矩阵特征值
        # m = np.real(AS).round(4).max()
        # all_nodes = new_G_small.nodes
        # #print(all_nodes)
        # da = {}                             ###!!!!!字典才对
        # for i in all_nodes:
        #     unfrozen_graph.remove_node(i)
        #     AS1 = nx.adjacency_spectrum(unfrozen_graph)
        #     m1 = np.real(AS1).round(4).max()
        #     da[i] = float(format(abs(m-m1)/m,'.4f'))   #单独运算看对不对
        #     unfrozen_graph = nx.Graph(frozen_graph)
        # dynage = max(da, key=lambda x: da[x]) 
        # # dynage=1
        # with open(filename_dynage,'a') as dynagef:
        #     dynagef.write(str(dynage))
        #     dynagef.write('\n')

        # nx.draw(new_G_small,with_labels = True)
        # plt.show()
    #np.savetxt('datatest.txt',adj_matrix)
    #print(adj_matrix.shape)
        countadj_now=adj_matrix.shape[1]
        countadj.append(adj_matrix.shape[1])
        countedge.append(Gnw.number_of_edges())
        # with open(filename_center,'a') as centerf:
        #     centerf.write(str(choice(Jordan_center)))    #Jordan center随机选，按理来说差别可能不大，实际有差别
        #     centerf.write('\n')
        # with open(filename_center1,'a') as centerf1:
        #     centerf1.write(str(Jordan_center[0]))    #Jordan center选第一个
        #     centerf1.write('\n')
        with open(filename_node_labels,'a') as labelf:#节点ID作为标签
            for k,v in resultall.items():
                labelf.write(str(v))
                labelf.write('\n')
        graph_labels.append(start_node)
        #graph_labels_class.append(partition[start_node])#所属的类
    else:
        adj_matrix=[]
        countadj_now=[]


    return (adj_matrix,countadj_now)

#可以自己造邻接矩阵，行和列的范围从adj_matrix.shape开始增加
#直接生成data_A.txt  边的邻接矩阵
def data_A(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_A = perfix + '_A.txt'
    filename_node_labels = perfix + '_dre_node_labels.txt'
    sum_ca_now = 0
    graphs=70
    nodehead=0
    nodetail=200
    with open(filename_A,'w') as f:
        with open(filename_node_labels,'w') as f1: #节点度记录
            #for nodesn in nodelist:
            for nodesn in range(nodehead,nodetail):
                for j in range(graphs):
                    adj,ca_now=genGraph(nodesn,datadir,bmname)
                    if len(adj):                      #图为非空，才进行下一步
                        coo_A=coo_matrix(adj)   #邻接矩阵的边的行/列的坐标
                        edge_index = [coo_A.row+1,coo_A.col+1]
                        #node_labels(adj)
                        a=np.array(adj)
                        a=np.sum(a,axis=1)
                        a=a.tolist()
                        for i in range(len(a)):
                            f1.write(str(a[i]))
                            f1.write('\n')
                        if len(countadj)==1:
                            for i in range(len(edge_index[1])):
                                f.write(str(edge_index[0][i])+','+str(edge_index[1][i]))
                                f.write('\n')
                                #print(str(coo_A.row[i])+','+str(coo_A.col[i]))
                        else:
                            for i in range(len(edge_index[1])):
                                f.write(str(edge_index[0][i]+sum_ca_now)+','+str(edge_index[1][i]+sum_ca_now))
                                f.write('\n')
                                #print(str(coo_A.row[i]+sum_ca_now)+','+str(coo_A.col[i]+sum_ca_now))
                        sum_ca_now=sum_ca_now+ca_now
    #/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data
    filename_readme = perfix + 'readme.txt'
    with open(filename_readme,'a') as f:
        #f.write('InfectionRate='+str(InfectionRate)+"\n")
        #f.write('Roundtime='+str(Roundtime)+"\n")
        f.write('[a,b]='+str(nodehead)+','+str(nodetail)+"\n")
        #f.write('nodelist='+str(nodelist)+'\n')
        f.write('every node graphs='+str(graphs)+"\n")


def main():

    bmname = 'BA200_p0.3_m70_n200'
    #path = os.path.join('/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data',bmname)
    path = os.path.join('/home/iot/zcy/usb/copy/myGCN/data/BA200/'+bmname+'/raw')
    
    #path = os.path.join('data',bmname)#调试时生成的文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    perfix = os.path.join(path,bmname)
    filename_readme = perfix+'readme.txt'
    with open(filename_readme,'w') as f:
        f.write('bmname = '+str(bmname)+"\n")
        f.write('N='+str(N)+"\n")
        f.write('底图='+fname+".npy"+"\n")
        f.write(fname+'底图，感染节点占part比例时停止传播，测试集，包括多种对比方法，分5类'+"\n")
        f.write('part='+str(part)+"\n")
        #f.write('val_datatest'+"\n")

    data=open(filename_readme,'a')
    data_A(path,bmname)
    graph_label(path,bmname)
    graph_indicator(path,bmname)
    #graph_label_classfication(path,bmname)

    dis_s=0
    for i in countedge:
        dis_s=dis_s+i
    print('sum of edges:',file=data)
    print(str(dis_s)+'\n',file=data)
    s1=0
    for i in countadj:
        s1=s1+i
    print('sum of adj:',file=data)
    print(str(s1)+'\n',file=data)

def graph_indicator(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_graph_indic = perfix + '_graph_indicator.txt'
    with open(filename_graph_indic,'w') as f:
        i=1
        for val in countadj:
            for j in range(int(val)):
                f.write(str(i))
                f.write('\n')
            i=i+1
def graph_label(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_graph_labels = perfix+ '_graph_labels.txt'
    with open(filename_graph_labels,'w') as f:
        for i in graph_labels:
            f.write(str(i))
            f.write('\n')
def graph_label_classfication(datadir,bmname): #分类后的节点标签
    perfix = os.path.join(datadir,bmname)
    filename_graph_labels_class = perfix+ '_graph_labels_class.txt'
    with open(filename_graph_labels_class,'w') as f:
        for i in graph_labels_class:
            f.write(str(i))
            f.write('\n')

if __name__ == "__main__":
    main()
