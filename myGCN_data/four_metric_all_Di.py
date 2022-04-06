#生成数据集来测试four_metric
from unittest import result
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
from PIL import Image
import cv2
#变量：B，sn范围,每个sn张数 ，Infectionrate, Roundtime，data文件名

N =67#记得必须改
fname = "high67"
G = nx.read_edgelist('./'+fname+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())

obn = int(N*0.5)  #有一半的节点可以观测到
observenode = random.sample(range(0,N-1),obn)#每次都随机选还是固定呢
#observenode= [387, 312, 379, 258, 17, 320, 33, 448, 302, 204, 405, 469, 34, 446, 418, 77, 426, 49, 109, 59, 487, 371, 158, 237, 140, 478, 473, 301, 470, 329, 122, 396, 333, 61, 173, 15, 374, 186, 452, 32, 150, 178, 261, 125, 27, 360, 251, 4, 318, 372, 93, 72, 28, 457, 128, 463, 114, 54, 149, 14, 462, 210, 113, 455, 141, 20, 321, 99, 388, 359, 351, 409, 291, 271, 227, 451, 143, 328, 284, 74, 304, 344, 496, 129, 289, 190, 306, 147, 468, 355, 89, 453, 429, 75, 94, 29, 184, 111, 9, 349, 299, 174, 376, 354, 416, 256, 13, 257, 265, 65, 249, 44, 217, 415, 461, 317, 481, 242, 253, 450, 364, 58, 489, 391, 76, 345, 454, 182, 16, 279, 402, 80, 222, 87, 12, 46, 471, 430, 300, 434, 248, 97, 439, 135, 8, 494, 126, 0, 96, 200, 81, 88, 212, 444, 123, 121, 493, 485, 230, 91, 73, 358, 214, 330, 127, 313, 267, 216, 180, 90, 474, 45, 384, 133, 235, 482, 266, 183, 366, 95, 175, 228, 460, 456, 124, 368, 472, 339, 153, 244, 309, 145, 335, 69, 232, 492, 419, 56, 138, 43, 283, 375, 268, 51, 63, 319, 377, 286, 433, 432, 262, 398, 479, 458, 105, 401, 103, 288, 132, 327, 169, 102, 334, 498, 119, 223, 356, 404, 19, 116, 336, 326, 62, 26, 315, 144, 269, 264, 278, 176, 193, 331, 118, 181, 423, 195, 495, 234, 310, 445]
#observenode= [i for i in range(1000)]
#observenode = [13, 8, 4, 33, 30, 7, 32, 15, 22, 28, 57, 12, 36, 44, 2, 38, 26, 43, 53, 34, 45, 3, 31, 35, 42, 29, 11, 5, 14, 6, 19] #BC 方法
#observenode = [13, 8, 4, 33, 30, 7, 32, 15, 22, 28, 57, 12, 36, 44, 2, 38, 26, 43, 53, 34, 45, 3, 31, 35, 42, 29, 11, 5, 14, 6, 19, 24, 21, 46, 1, 0, 47, 18, 9, 52, 17, 16, 41] #BC,OB=0.7
#observenode = [13, 8, 4, 33, 30, 7, 32, 15, 22, 28, 57, 12, 36, 44, 2, 38, 26, 43] #0.3
print(obn,observenode)
partition=np.load(fname+'_p'+'.npy',allow_pickle = True).item()
graph_labels = []
graph_labels_class=[]
countedge = []
countadj = []
part=0.1
InfectionRate = 0.5#概率太大，10轮感染1400个节点
Roundtime = 3
adjall = []
def four_metirc(sG,negative):
    #1#能到达positive数量最多的节点
    center_frozen_graph = nx.freeze(sG)#G被冷冻为frozen_graph，不会改变
    center_unfrozen_graph = nx.Graph(center_frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
    Gc_node = max(nx.weakly_connected_components(sG), key=len)   #sub_newG的最大连通子图来求Jordan Center     可能为空图
    #print(Gc_node)
    Gc = center_unfrozen_graph.subgraph(Gc_node)
    #2取距离之和最小的点
    lengths = nx.all_pairs_dijkstra_path_length(Gc,weight='weight')
    lengths = dict(lengths)
    ec={}#每个节点距离其他节点的距离之和
    for ei in lengths:       #当观测图的编号不从0按顺序开始时；
        ec[ei]=sum(lengths[ei].values())
    res=[]
    for x,v in ec.items():
        if v == min(ec.values()):
            res.append(x)
    # print(ec)
    # print(res)
    #3不能到达negative的节点数量最多的点
    #4与negative的距离之和，取最大的点
    read_dic = np.load(fname+"_short_path.npy",allow_pickle = True).item()
    four_dis = {}
    for i in res:
        dis_ne = 0
        for j in negative:
            dis_ne = dis_ne+read_dic[i][j]
        four_dis[i]=dis_ne
    #print("four_dis:",four_dis)
    result = min(four_dis, key = lambda x:four_dis[x])
    #print("result:",result)
    return result

def genGraph(sn,datadir,bmname,m):

#感染过程
    node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
    S = node
    I = []
    negative = []
    positive =[]
    j=0
    while j<1:
        #start_node = random.choice(node)#1个初始感染节点
        start_node = sn#1个初始感染节点
        I.append(start_node)
        S.remove(start_node)
        j=j+1
    
    print(start_node)
    
    new_G_small = nx.DiGraph()    ###感染图
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
                            new_G_small.add_edge(int(nbr),int(key))     #I->S
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
    print(len(G.nodes()))
    print(len(G.edges()))
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
    perfix = os.path.join(datadir,bmname)
    #filename_node_labels = perfix + '_xnode_labels.txt'
    filename_b = perfix + '_four_metric_BC_b.txt'
    filename_m = perfix + '_four_metric_BC_m.txt'
    filename_w = perfix + '_four_metric_BC_w.txt'
    #filename_center1 = perfix + '_jordancenter_w.txt'
    #filename_adj = perfix+'_adjdata'
    # filename_unbet = perfix + '_unbet.txt'
    # filename_discen = perfix + '_discen.txt'
    # filename_dynage = perfix + '_dynage.txt'
    #new_G_small表示感染图
    #new_G表示感染图加上未感染的节点，
    #将new_G中的邻接矩阵扩展到G
    new_G = nx.DiGraph()
    new_G.add_nodes_from(i for i in range(N))
    new_G.add_nodes_from(new_G_small.nodes())    #不增加单独节点看实验效果如何，max_nodes=100时，max graph size 是否为100
    new_G.add_edges_from(new_G_small.edges(data=True))
    #Jordan_center  = nx.center(new_G)#非全连接图不能计算

    if not nx.is_empty(subgraph):   #进入到four_metric中的图不要为空图
        adj_matrix = nx.adjacency_matrix(new_G).todense()
        ###Jordan_center  = nx.center(new_G_small)    #不考虑权重的jc

        countadj_now=adj_matrix.shape[1]
        countadj.append(adj_matrix.shape[1])
        countedge.append(new_G.number_of_edges())
        result_b = four_metirc(sub_new_G_b,negative)
        with open(filename_b,'a') as centerf:
            centerf.write(str(result_b))    #Jordan center
            centerf.write('\n')
        result_m = four_metirc(sub_new_G_m,negative)
        with open(filename_m,'a') as centerm:
            centerm.write(str(result_m))    #Jordan center
            centerm.write('\n')
        result_w = four_metirc(sub_new_G_w,negative)
        with open(filename_w,'a') as centerw:
            centerw.write(str(result_w))    #Jordan center
            centerw.write('\n')
        # with open(filename_node_labels,'a') as labelf:#节点ID作为标签
        #     for i in new_G.nodes:
        #         labelf.write(str(i))
        #         labelf.write('\n')
        graph_labels.append(start_node)
        #graph_labels_class.append(start_node//100)#所属的类   0-99的分类方法
        graph_labels_class.append(partition[start_node])####k-means  反而没；分类方法
    else:
        adj_matrix=[]
        countadj_now=[]
    # A = nx.to_numpy_matrix(new_G) 
    # adjall.append(A)
    # if not os.path.exists(filename_adj):
    #     os.makedirs(filename_adj)
    # number = sn*100+m
    # cv2.imwrite(filename_adj+'/'+str(number)+'.png',A)
    #####
    # im = Image.fromarray(A)
    # im.convert('1').save(filename_adj+'/'+str(sn)+'.jpeg')
    return (adj_matrix,countadj_now)

#可以自己造邻接矩阵，行和列的范围从adj_matrix.shape开始增加
#直接生成data_A.txt  边的邻接矩阵
def data_A(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_A = perfix + '_A.txt'
    filename_node_labels = perfix + '_dre_node_labels.txt'
    sum_ca_now = 0
    graphs=20
    nodehead=0
    nodetail=67
    # class0= [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 24, 25, 26, 28, 31, 43, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 75, 76, 78, 80, 82, 83, 88, 89, 92, 93, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 107, 109, 144, 166, 170, 172, 174, 175, 177, 178, 180, 181, 182, 183, 184, 186, 188, 202, 203, 205, 207, 208, 211, 212, 213, 214, 215, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 244, 245, 246, 249, 258, 260, 261, 262, 263, 265, 268, 271, 273, 274, 275, 276, 286, 290, 291, 296, 298, 310, 326, 327, 328, 329, 330, 335, 345, 346, 347, 348, 363, 364, 368, 372, 374, 381, 382, 384, 385, 386, 387, 390, 393, 395, 396, 398, 402, 403, 411, 413, 415, 423, 424, 425, 426, 427, 428, 429, 430, 432, 435, 440, 450, 451, 455, 461, 468, 469, 470, 471, 474, 478, 479, 481, 482, 485, 487, 495]
    # class1= [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 64, 69, 74, 77, 79, 85, 99, 102, 110, 111, 134, 138, 161, 162, 163, 164, 167, 168, 169, 171, 173, 191, 192, 193, 194, 198, 204, 247, 252, 256, 259, 264, 267, 269, 270, 272, 281, 282, 283, 292, 293, 294, 299, 302, 303, 304, 305, 306, 315, 323, 325, 349, 350, 367, 369, 370, 371, 376, 377, 383, 394, 401, 416, 417, 418, 419, 420, 421, 449, 452, 453, 454, 458, 459, 473, 475, 476, 486, 490, 493]
    # class2= [33, 34, 35, 36, 37, 38, 49, 112, 113, 116, 118, 119, 120, 121, 122, 124, 126, 127, 129, 130, 131, 132, 135, 136, 137, 139, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 189, 216, 217, 218, 254, 255, 277, 278, 288, 300, 301, 308, 309, 311, 312, 313, 314, 317, 318, 319, 336, 337, 338, 339, 343, 344, 373, 378, 379, 380, 388, 389, 397, 399, 400, 404, 405, 406, 407, 412, 437, 462, 463, 464, 465, 480, 483, 488, 489, 492, 494, 496, 497, 498, 499]
    # class3= [8, 10, 12, 27, 29, 30, 32, 39, 40, 41, 42, 44, 45, 81, 84, 86, 87, 90, 91, 106, 108, 114, 115, 117, 123, 125, 128, 133, 140, 141, 142, 143, 165, 176, 179, 185, 187, 190, 195, 196, 197, 199, 200, 201, 206, 209, 210, 219, 220, 221, 222, 238, 241, 242, 243, 248, 250, 251, 253, 257, 266, 279, 280, 284, 285, 287, 289, 295, 297, 307, 316, 320, 321, 322, 324, 331, 332, 333, 334, 340, 341, 342, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 365, 366, 375, 391, 392, 408, 409, 410, 414, 422, 431, 433, 434, 436, 438, 439, 441, 442, 443, 444, 445, 446, 447, 448, 456, 457, 460, 466, 467, 472, 477, 484, 491]
    class0= [33, 34, 35, 36, 37, 38, 49, 112, 113, 116, 118, 119, 120, 121, 122, 124, 126, 127, 129, 130, 131, 132, 135, 136, 137, 139, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 189, 216, 217, 218, 254, 255, 277, 278, 288, 300, 301, 308, 309, 311, 312, 313, 314, 317, 318, 319, 336, 337, 338, 339, 343, 344, 373, 378, 379, 380, 388, 389, 397, 399, 400, 404, 405, 406, 407, 412, 437, 462, 463, 464, 465, 480, 483, 488, 489, 492, 494, 496, 497, 498, 499]
    class1= [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 64, 69, 74, 77, 79, 85, 99, 102, 110, 111, 134, 138, 161, 162, 163, 164, 167, 168, 169, 171, 173, 191, 192, 193, 194, 198, 204, 247, 252, 256, 259, 264, 267, 269, 270, 272, 281, 282, 283, 292, 293, 294, 299, 302, 303, 304, 305, 306, 315, 323, 325, 349, 350, 367, 369, 370, 371, 376, 377, 383, 394, 401, 416, 417, 418, 419, 420, 421, 449, 452, 453, 454, 458, 459, 473, 475, 476, 486, 490, 493]
    class2= [0, 1, 2, 4, 5, 6, 31, 43, 46, 47, 48, 50, 55, 57, 58, 70, 71, 75, 76, 78, 172, 180, 183, 188, 202, 203, 207, 223, 225, 226, 227, 229, 230, 231, 236, 239, 244, 245, 249, 310, 326, 327, 328, 329, 330, 345, 346, 347, 348, 363, 368, 372, 382, 384, 387, 396, 398, 402, 403, 413, 415, 432, 435, 450, 451, 455, 469, 471, 481, 487]
    class3= [3, 7, 9, 11, 12, 24, 25, 26, 27, 28, 32, 51, 52, 53, 54, 56, 59, 60, 61, 62, 63, 65, 66, 67, 68, 72, 73, 80, 82, 83, 88, 89, 92, 93, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 107, 108, 109, 144, 165, 166, 170, 174, 175, 176, 177, 178, 181, 182, 184, 186, 205, 206, 208, 211, 212, 213, 214, 215, 224, 228, 232, 233, 234, 235, 237, 238, 240, 241, 246, 258, 260, 261, 262, 263, 265, 268, 271, 273, 274, 275, 276, 279, 280, 284, 286, 287, 290, 291, 296, 297, 298, 316, 324, 331, 332, 333, 335, 352, 353, 356, 361, 362, 364, 374, 381, 385, 386, 390, 393, 395, 408, 409, 410, 411, 423, 424, 425, 426, 427, 428, 429, 430, 436, 440, 447, 460, 461, 468, 470, 472, 474, 478, 479, 482, 485, 495]
    class4= [8, 10, 29, 30, 39, 40, 41, 42, 44, 45, 81, 84, 86, 87, 90, 91, 106, 114, 115, 117, 123, 125, 128, 133, 140, 141, 142, 143, 179, 185, 187, 190, 195, 196, 197, 199, 200, 201, 209, 210, 219, 220, 221, 222, 242, 243, 248, 250, 251, 253, 257, 266, 285, 289, 295, 307, 320, 321, 322, 334, 340, 341, 342, 351, 354, 355, 357, 358, 359, 360, 365, 366, 375, 391, 392, 414, 422, 431, 433, 434, 438, 439, 441, 442, 443, 444, 445, 446, 448, 456, 457, 466, 467, 477, 484, 491]
    ###97 , 91 , 70 , 146 , 96
    #nodelist = class1

    #for nodesn in nodelist:
    for nodesn in range(nodehead,nodetail):
        for j in range(graphs):
            adj,ca_now=genGraph(nodesn,datadir,bmname,j)


    # with open(filename_A,'w') as f:
    #     with open(filename_node_labels,'w') as f1: #节点度记录
    #         for nodesn in nodelist:
    #         #for nodesn in range(nodehead,nodetail):
    #             for j in range(graphs):
    #                 adj,ca_now=genGraph(nodesn,datadir,bmname,j)
    #                 if len(adj):                      #图为非空，才进行下一步
    #                     coo_A=coo_matrix(adj)   #邻接矩阵的边的行/列的坐标
    #                     edge_index = [coo_A.row,coo_A.col]
    #                     #node_labels(adj)
    #                     a=np.array(adj)
    #                     a=np.sum(a,axis=1)
    #                     a=a.tolist()
    #                     for i in range(len(a)):
    #                         f1.write(str(a[i]))
    #                         f1.write('\n')
    #                     if len(countadj)==1:
    #                         for i in range(len(edge_index[1])):
    #                             f.write(str(coo_A.row[i])+','+str(coo_A.col[i]))
    #                             f.write('\n')
    #                             #print(str(coo_A.row[i])+','+str(coo_A.col[i]))
    #                     else:
    #                         for i in range(len(edge_index[1])):
    #                             f.write(str(coo_A.row[i]+sum_ca_now)+','+str(coo_A.col[i]+sum_ca_now))
    #                             f.write('\n')
    #                             #print(str(coo_A.row[i]+sum_ca_now)+','+str(coo_A.col[i]+sum_ca_now))
    #                     sum_ca_now=sum_ca_now+ca_now
    #/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data
    filename_readme = perfix + 'readme.txt'
    with open(filename_readme,'a') as f:
        #f.write('InfectionRate='+str(InfectionRate)+"\n")
        #f.write('Roundtime='+str(Roundtime)+"\n")
        f.write('[a,b]='+str(nodehead)+','+str(nodetail)+"\n")
        #f.write('nodelist='+str(nodelist)+'\n')
        f.write('every node graphs='+str(graphs)+"\n")


def main():

    bmname = 'high67_p0.1_ob0.5_fm_m20_BC'
    #path = os.path.join('/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data',bmname)
    #path = os.path.join('/home/iot/zcy/usb/copy/rexying_diffpool/diffpool-master/data',bmname)
    #path = os.path.join('/home/iot/zcy/usb/copy/MINIST/MINIST/mydata',bmname)
    path = os.path.join('/home/iot/zcy/usb/copy/research/graph_python/research/2.25_Di/data',bmname)
    
    #path = os.path.join('data',bmname)#调试时生成的文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    perfix = os.path.join(path,bmname)
    filename_readme = perfix+'readme.txt'
    with open(filename_readme,'w') as f:
        f.write('bmname = '+str(bmname)+"\n")
        f.write('N='+str(N)+"\n")
        f.write('底图='+fname+".npy"+"\n")
        f.write(fname+'底图，感染节点占part比例时停止传播，z=0.1，包括多种对比方法，平均分5类'+"\n")
        f.write('part='+str(part)+"\n")
        #f.write('val_datatest'+"\n")

    data=open(filename_readme,'a')
    data_A(path,bmname)
    graph_label(path,bmname)
    graph_indicator(path,bmname)
    graph_label_classfication(path,bmname)
    #np.save(perfix+ '_adj',adjall)
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
    # filename_adj = perfix+'_adj.txt'
    # with open(filename_adj,'a') as fadj:
    #     for i in range(10):
    #         for j in range(N):
    #             fadj.write(str(adjall[i][j]))
    #             fadj.write('\n')

if __name__ == "__main__":
    main()
