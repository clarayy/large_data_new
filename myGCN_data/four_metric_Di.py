#传感器图，four_metric方法

#新的感染图画法，添加边根据所有感染节点来增加
#propagation_pro1.py的升级
#给边增加权重，随机增加
#改为传播到总结点数的0.8停止传播
#propagation使用的是从I开始遍历，使S改变状态
#本程序从S开始遍历，感染概率为1-（1-q）^n
#在一张图上同时显示初始图结构和传播感染图，即传播感染图是初始图结构的一部分
#from research.new_propagation_SI import InfectionRate, Roundtime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx
import os
import random
from itertools import islice
import argparse

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
N =67#记得必须改
fname = "high67"
G = nx.read_edgelist('./'+fname+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())

part=0.1
Roundtime = 3

obn = int(N*0.5)  #有一半的节点可以观测到
observenode = random.sample(range(0,N-1),obn)#每次都随机选还是固定呢
#observenode =[1, 4, 5, 6, 11, 12, 13, 15, 17, 18, 19, 22, 25, 26, 27, 28, 29, 32, 33, 35, 39, 41, 43, 45, 48, 49, 50, 52, 58, 59, 60]
#observenode= [i for i in range(62)]
print(observenode)
######BC求ob
observenode=[]
bet_cen = nx.betweenness_centrality(G)#节点的中介中心性
bet_cen = sorted(bet_cen.items(),key = lambda x :x[1],reverse=True)
for i in range(obn):
    observenode.append(bet_cen[i][0])
#print(observenode)
#[13, 8, 4, 33, 30, 7, 32, 15, 22, 28, 57, 12, 36, 44, 2, 38, 26, 43, 53, 34, 45, 3, 31, 35, 42, 29, 11, 5, 14, 6, 19]
# #####NI方法
# observenode=[]
# deg = G.degree()#节点的度
# deg = sorted(deg,key = lambda x :x[1],reverse=True)
# for i in range(obn):
#     observenode.append(deg[i][0])
# #print(observenode)
# #[2, 33, 34, 38, 22, 36, 32, 7, 28, 4, 43, 8, 26, 3, 44, 45, 47, 13, 24, 15, 1, 5, 6, 40, 41, 29, 19, 52, 0, 42, 21]
# #print(deg)

print(obn,observenode)

#感染过程
node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
#node1 = list(set(node))#节点元素从小到大排序
#print(len(node))
S = node
I = []
negative = []
positive =[]
j=0
while j<1:
    #start_node = random.choice(node)#1个初始感染节点
    start_node = 18#1个初始感染节点
    I.append(start_node)
    S.remove(start_node)
    j=j+1
print("start_node:")
print(start_node)
new_G_A = nx.DiGraph()    ###感染图
sub_new_G_b = nx.DiGraph()   ##观测到的感染图
sub_new_G_w = nx.DiGraph()   ##观测到的感染图
sub_new_G_m = nx.DiGraph() 
count = [1]
statechange = []
edgechange = []
edgeweight = []
weight_s = 1
#for r in range(Roundtime):       #####从I开始遍历，小于边的概率就感染
while len(I)<=part*len(G.nodes()):
    for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
        if int(nbr) in I:            
            for key in datadict:
                if int(key) in S:
                    rate = G.get_edge_data(int(nbr),int(key))['weight']   #I->S 
                    if random.random() <= rate: 
                        statechange.append(int(key))
                        new_G_A.add_edge(int(nbr),int(key))     #I->S
    statechange=list(set(statechange))
    for i in statechange:
        S.remove(i)
        I.append(i)
    if len(I)==1:
        break
    count.append(len(I))
    statechange = []
ob_I =[]
for gan in observenode:
    if gan in I:
        ob_I.append(gan)
subgraph = G.subgraph(ob_I)
for u,v,w in subgraph.edges(data=True):
    #print(w)
    sub_new_G_b.add_edge(u,v,weight = round(w['weight'],2))
    sub_new_G_m.add_edge(u,v,weight = 1)
    sub_new_G_w.add_edge(u,v,weight = round(1-w['weight'],2))
for i in observenode:
    if i in I:
        positive.append(i)
    else:
        negative.append(i)
positive =list(set(positive))
print("positive:",positive)
print("negative:",negative)
if not nx.is_empty(new_G_A):
    #1#能到达positive数量最多的节点
    center_frozen_graph = nx.freeze(sub_new_G_m)#G被冷冻为frozen_graph，不会改变
    center_unfrozen_graph = nx.Graph(center_frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
    Gc_node = max(nx.weakly_connected_components(sub_new_G_m), key=len)   #sub_newG的最大连通子图来求Jordan Center     可能为空图
    print("Gc_node:",Gc_node)
    Gc = center_unfrozen_graph.subgraph(Gc_node)
    #2取距离之和最小的点
    lengths = nx.all_pairs_dijkstra_path_length(Gc,weight='weight')
    lengths = dict(lengths)
    ec={}#每个节点距离其他节点的距离之和
    for ei in lengths:       #当观测图的编号不从0按顺序开始时；
        ec[ei]=sum(lengths[ei].values())
    res=[]   #距离之和的最小值对应的节点
    for x,v in ec.items():
        if v == min(ec.values()):    #距离之和的最小值
            res.append(x)
    print("ec:",ec)
    print("res:",res)
    #3不能到达negative的节点数量最多的点（都能到达）
    #4与negative的距离之和，取最大的点
    read_dic = np.load(fname+"_short_path.npy",allow_pickle = True).item()
    four_dis = {}    #第二步得到的节点与negative的距离
    for i in res:
        dis_ne = 0
        for j in negative:
            dis_ne = dis_ne+read_dic[i][j]
        four_dis[i]=dis_ne
    print("four_dis:",four_dis)
    result = max(four_dis, key = lambda x:four_dis[x])   #距离之和的最大值
    print("result:",result)

plt.figure()
Innum = plt.subplot(111)
props = {'title':'Total number of Infected Users',
          'ylabel':'number','xlabel':'time'}
Innum.set(**props)
x = []
for i in range(0,len(count)):
    x.append(i)
Innum.plot(x,count)
#Innum.text(0,0,'N:%.0f,Rate:%.4f' %(N,InfectionRate))
Innum.text(0,0,'N:%.0f' %N)
print("I=",I)
print('len(I):')
print(len(I))
plt.show()  #显示感染曲线

#nx.draw(new_G)
pos = nx.spring_layout(sub_new_G_m) #为什么报错！！！！！！！！！！
nx.draw(sub_new_G_m,pos,with_labels = True)#逗号是中文的  #画图
plt.show()
# pos = nx.spring_layout(G)
# nx.draw(G,pos,node_color='b',node_size=1,edge_color = 'b',with_labels=True)
# plt.show()