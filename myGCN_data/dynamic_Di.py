#SI模型，完全观测，dynamic-ages；
#邻接矩阵考虑权重，结果不同

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

N =1000
name = "BA1000"
# B = np.load(name+".npy")#读取固定的图
# G = nx.Graph(B)
G = nx.read_edgelist('./'+name+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())
# for u,v in G.edges():
#     G.add_edge(u,v,weight=random.random())
# print(G.edges(data=True))
# pos=nx.spring_layout(G)
# #print(G.edges(data=True))
# nx.draw(G,pos,node_size=1,with_labels = True)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
# plt.show()
InfectionRate = 0.1#概率太大，10轮感染1400个节点
Roundtime =3
'''
#测试用的小图
N = 34
filename = '/home/zhang/Documents/research/graph_python/research/graphtest.txt'
G = nx.Graph()
with open(filename,'r') as file:
    for line in file:
        head,tail=[str(x) for x in line.split( )]
        G.add_edge(head,tail)
#nx.draw(G,with_labels = True)
#plt.show()
'''

#感染过程
node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
#node1 = list(set(node))#节点元素从小到大排序
#print(len(node))
S = node
I = []

j=0
while j<1:
    #start_node = random.choice(node)#1个初始感染节点
    start_node = 499#1个初始感染节点
    I.append(start_node)
    S.remove(start_node)
    j=j+1
print("start_node:")
print(start_node)
part = 0.1
sub_A = nx.DiGraph()#感染过程中逐渐添加边形成的图
sub_B = nx.DiGraph()#所有I节点连接形成的图
new_G_b = nx.DiGraph()#best_weight
new_G_m = nx.DiGraph()#yuan_weight
new_G_w = nx.DiGraph()#worst_weight
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
                        sub_A.add_edge(int(nbr),int(key))     #I->S
    statechange=list(set(statechange))
    for i in statechange:
        S.remove(i)
        I.append(i)

    count.append(len(I))
    statechange = []

# pos = nx.spring_layout(sub_A) #为什么报错！！！！！！！！！！
# nx.draw(sub_A,pos,with_labels = True)#逗号是中文的  #画图
# plt.show()
# print("len(sub__A.edges()):",len(sub_A.edges()))
# print("I:",I)
# print("len(I):",len(I))
sub_B = G.subgraph(I)
pos = nx.spring_layout(sub_B) #为什么报错！！！！！！！！！！
nx.draw(sub_B,pos,with_labels = True)#逗号是中文的  #画图
plt.show()
print("len(sub_B.edges())",len(sub_B.edges(data=True)))
#new_G_b=sub_B
# new_G_b.add_nodes_from(sub_B.nodes())
# new_G_b.add_edges_from(sub_B.edges(data=True))

for u,v,w in sub_B.edges(data=True):
    #print(w)
    new_G_b.add_edge(u,v,weight = round(w['weight'],2))
    new_G_m.add_edge(u,v,weight = 1)
    new_G_w.add_edge(u,v,weight = round(1-w['weight'],2))
#print(new_G_b.edges(data=True))
print("len(new_G_b.edges()):",len(new_G_b.edges()))
#print(G.edges())
plt.figure()
Innum = plt.subplot(111)
props = {'title':'Total number of Infected Users',
          'ylabel':'number','xlabel':'time'}
Innum.set(**props)
x = []
for i in range(0,len(count)):
    x.append(i)
Innum.plot(x,count)
Innum.text(0,0,'N:%.0f,Rate:%.4f' %(N,InfectionRate))
print('len(I):',len(I))
print("I=",I)
plt.show()  #显示感染曲线

#显示源节点
#存储边数据
#nx.write_edgelist(new_G,'data_G1.txt',delimiter=',',data = False)
#print(new_G.edges())
#print(len(new_G.edges()))
print('len(new_G_b.nodes()):')
print(len(new_G_b.nodes()))
# print('edges:')
# print(new_G.edges())
print('len(new_G_m.nodes()):')
print(len(new_G_m.nodes()))
# print('new_G_duizhao edges:')
# print(new_G_duizhao.edges())
#nx.draw(new_G)
pos = nx.spring_layout(new_G_b) #为什么报错！！！！！！！！！！
nx.draw(new_G_b,pos,with_labels = True)#逗号是中文的  #画图
labels = nx.get_edge_attributes(new_G_b,'weight')
nx.draw_networkx_edge_labels(new_G_b, pos, edge_labels = labels)
plt.show()
A = nx.to_numpy_matrix(new_G_b) 
print(A)
print(A.shape)
pos = nx.spring_layout(new_G_m) #为什么报错！！！！！！！！！！
nx.draw(new_G_m,pos,with_labels = True)#逗号是中文的  #画图
labels = nx.get_edge_attributes(new_G_m,'weight')
nx.draw_networkx_edge_labels(new_G_m, pos, edge_labels = labels)
plt.show()
A = nx.to_numpy_matrix(new_G_m) 
print(A)
print(A.shape)
pos = nx.spring_layout(new_G_w) #为什么报错！！！！！！！！！！
nx.draw(new_G_w,pos,with_labels = True)#逗号是中文的  #画图
labels = nx.get_edge_attributes(new_G_w,'weight')
nx.draw_networkx_edge_labels(new_G_w, pos, edge_labels = labels)
plt.show()
A = nx.to_numpy_matrix(new_G_w) 
print(A)
print(A.shape)
#dynamic ages
def dynamic_ages(G):
    frozen_graph = nx.freeze(G)#G被冷冻为frozen_graph，不会改变
    unfrozen_graph = nx.Graph(frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
    AS = nx.adjacency_spectrum(frozen_graph)#邻接矩阵特征值
    m = np.real(AS).round(4).max()
    all_nodes = G.nodes
    #print(all_nodes)
    da = {}                             ###!!!!!字典才对
    for i in all_nodes:
        unfrozen_graph.remove_node(i)
        AS1 = nx.adjacency_spectrum(unfrozen_graph)
        m1 = np.real(AS1).round(4).max()
        da[i] = float(format(abs(m-m1)/m,'.4f'))   #单独运算看对不对
        unfrozen_graph = nx.Graph(frozen_graph)
    dynage = max(da, key=lambda x: da[x]) 
    #print("dynage:",dynage)
    return dynage
print("new_G_b_dynage:",dynamic_ages(new_G_b))
print("new_G_m_dynage:",dynamic_ages(new_G_m))
print("new_G_w_dynage:",dynamic_ages(new_G_w))
# with open(filename_dynage,'a') as dynagef:
#     dynagef.write(str(dynage))
#     dynagef.write('\n')
#在一张图上显示
Gt=new_G_b.copy()
for node in I:
    if node!= start_node:
        Gt.remove_node(node)
#print(len(Gt.nodes()))#源节点
#print(len(G.nodes()))
pos = nx.spring_layout(G)
print("len(G.edges():",len(G.edges()))
# nx.draw(G,pos,node_color='b',node_size=1,edge_color = 'b',with_labels=True)
# nx.draw_networkx_nodes(new_G_b,pos,node_color='r',node_size=1)
# nx.draw_networkx_edges(new_G_b,pos,edge_color = 'r')
# nx.draw(Gt,pos,node_color = 'red',node_size = 10)
# plt.show()
