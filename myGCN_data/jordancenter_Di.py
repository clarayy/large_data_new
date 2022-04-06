#部分观测且jordan center
#ob_B_SI.py+weight_jordancenter.py
from hashlib import new
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
N =2068#记得必须改
fname = "p2p2068"
G = nx.read_edgelist('./'+fname+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())

part=0.1
Roundtime = 3

obn = int(N*0.5)  #有一半的节点可以观测到
observenode = random.sample(range(0,N-1),obn)#每次都随机选还是固定呢
#observenode= [i for i in range(62)]
#1034个，0.5
#observenode =[1186, 1512, 317, 518, 587, 947, 1293, 475, 954, 528, 1975, 1142, 1465, 881, 1934, 1270, 1335, 1014, 202, 1765, 596, 1103, 278, 575, 1488, 366, 1636, 1337, 1369, 892, 33, 608, 1372, 707, 28, 1295, 1511, 1820, 129, 110, 1081, 298, 143, 1933, 1905, 49, 916, 1220, 1507, 1818, 1227, 86, 1239, 373, 1223, 1333, 1556, 1883, 1013, 1349, 90, 1790, 1028, 1170, 889, 1284, 1942, 1032, 1860, 1579, 208, 1370, 1380, 1830, 499, 1691, 1525, 1033, 497, 1137, 200, 868, 1682, 853, 1639, 259, 1047, 341, 1969, 1943, 1720, 1114, 966, 1443, 1439, 36, 1757, 784, 9, 856, 1444, 1150, 455, 50, 797, 718, 193, 359, 1494, 1797, 1768, 720, 292, 1984, 2061, 1965, 1709, 595, 377, 526, 1799, 135, 1540, 1603, 712, 634, 1839, 803, 380, 1909, 1615, 1225, 1496, 1089, 137, 1266, 85, 1168, 1086, 1745, 1696, 600, 375, 530, 1856, 1123, 303, 1573, 897, 1107, 1544, 839, 1083, 318, 658, 1551, 1012, 1476, 329, 1643, 274, 1185, 1029, 209, 1212, 282, 1283, 1265, 1037, 390, 1079, 1850, 1394, 42, 182, 165, 1, 1779, 1541, 990, 753, 1652, 1358, 1061, 748, 1007, 2052, 457, 936, 1242, 891, 234, 170, 1554, 1434, 555, 1200, 262, 1245, 1618, 563, 1165, 1240, 1657, 448, 665, 231, 1734, 1697, 1957, 212, 975, 408, 1514, 175, 1536, 1641, 654, 833, 1463, 1590, 1888, 549, 179, 239, 1840, 237, 12, 1819, 896, 1994, 770, 1753, 1063, 89, 811, 1600, 1020, 791, 1769, 819, 1623, 604, 1183, 2060, 334, 178, 505, 2037, 1411, 440, 1216, 583, 1248, 112, 1346, 1436, 51, 1487, 645, 2013, 283, 907, 1461, 160, 2009, 71, 504, 439, 1827, 854, 633, 862, 301, 410, 1299, 567, 312, 588, 1608, 265, 1726, 1872, 127, 114, 794, 1218, 65, 1045, 52, 723, 848, 1822, 2064, 1111, 517, 1254, 997, 1712, 1740, 999, 1962, 693, 1365, 1147, 700, 17, 1446, 1728, 817, 1052, 992, 1261, 737, 1863, 885, 678, 1901, 1793, 1530, 1330, 531, 1141, 1060, 1347, 1087, 903, 319, 479, 741, 1602, 321, 37, 824, 1157, 1847, 2035, 1450, 22, 1221, 40, 2015, 1966, 108, 878, 1320, 1255, 1210, 1900, 72, 804, 1801, 1305, 627, 1891, 1504, 1398, 1800, 1096, 1036, 1606, 415, 1445, 1191, 698, 1314, 482, 796, 476, 473, 462, 8, 398, 1042, 1742, 844, 676, 171, 145, 1076, 1490, 1056, 1474, 1193, 75, 682, 354, 1279, 249, 681, 347, 827, 1549, 1781, 1139, 158, 1430, 1518, 2030, 910, 1342, 1482, 697, 673, 295, 565, 405, 131, 1585, 1478, 1588, 2063, 1583, 873, 1843, 146, 1238, 1515, 524, 1055, 1297, 1278, 1795, 1077, 913, 1194, 397, 1532, 503, 1923, 906, 846, 1772, 1135, 901, 1322, 948, 58, 1955, 1698, 995, 1022, 1692, 511, 1998, 261, 618, 1415, 1048, 1694, 732, 789, 142, 535, 2011, 762, 168, 402, 1484, 1237, 139, 1309, 1927, 426, 973, 1481, 1432, 1058, 981, 263, 1690, 852, 964, 360, 830, 1164, 1122, 1842, 1146, 2062, 1353, 34, 1198, 1026, 11, 1570, 116, 692, 1695, 284, 1290, 1030, 1952, 1379, 38, 1117, 1862, 754, 540, 1941, 128, 1892, 601, 798, 579, 506, 1205, 1710, 1687, 1936, 478, 205, 140, 1339, 1853, 392, 21, 260, 1813, 1881, 1659, 1704, 220, 1522, 1009, 1187, 648, 1418, 615, 1470, 1112, 53, 10, 1852, 1711, 1519, 96, 1786, 1455, 1981, 598, 1756, 1046, 281, 311, 1485, 1499, 1634, 1252, 1724, 1526, 822, 1095, 1040, 424, 823, 1632, 1501, 1589, 1121, 1644, 1973, 2034, 1910, 388, 1846, 270, 636, 1595, 235, 496, 328, 1130, 1917, 2028, 1919, 1838, 404, 2065, 1908, 19, 445, 180, 454, 1880, 972, 1257, 1817, 1390, 1421, 1497, 620, 1491, 581, 1301, 1723, 1419, 31, 1479, 1597, 324, 1560, 1325, 1049, 176, 508, 1620, 2057, 299, 1832, 1961, 1715, 1539, 1895, 1542, 1587, 1744, 1705, 177, 985, 1738, 905, 1596, 306, 1949, 1561, 1855, 1367, 215, 18, 1671, 130, 1323, 1626, 1783, 1374, 1986, 267, 1214, 1395, 578, 509, 1378, 1747, 1766, 490, 1368, 427, 483, 1410, 626, 339, 115, 1156, 1300, 1384, 810, 486, 1231, 78, 228, 610, 1650, 1782, 157, 2041, 1480, 778, 76, 376, 226, 1834, 751, 1409, 1016, 758, 201, 327, 1859, 617, 1586, 807, 2038, 195, 1088, 694, 450, 325, 886, 453, 546, 2024, 937, 736, 251, 1506, 1565, 230, 361, 1977, 84, 1654, 1246, 727, 1133, 484, 1093, 895, 1172, 1967, 624, 1228, 552, 254, 980, 310, 1025, 1667, 1215, 1038, 1206, 1006, 1857, 409, 393, 1420, 719, 1069, 162, 952, 257, 777, 1331, 156, 904, 551, 1105, 104, 1281, 708, 126, 300, 1836, 1611, 688, 1054, 1356, 2066, 1684, 1031, 2016, 1457, 1178, 650, 183, 400, 181, 349, 942, 481, 1155, 1527, 799, 238, 1775, 1023, 1199, 1889, 1845, 429, 501, 745, 2040, 863, 1867, 1646, 793, 1145, 1068, 1815, 1665, 364, 247, 570, 522, 1454, 1613, 1475, 1735, 119, 1548, 1066, 1828, 1811, 1127, 519, 1201, 1686, 252, 1468, 98, 663, 630, 640, 1329, 29, 1458, 1236, 1163, 1412, 1708, 874, 2048, 912, 1581, 396, 711, 674, 1005, 1950, 899, 1288, 256, 13, 838, 1979, 1431, 141, 1821, 761, 443, 1806, 1180, 782, 93, 1610, 938, 523, 1971, 619, 781, 1070, 738, 258, 2055, 203, 713, 64, 1673, 99, 821, 840, 2006, 1184, 1074, 1272, 365, 342, 514, 1241, 289, 304, 1762, 315, 113, 485, 744, 1904, 515, 1778, 1714, 704, 356, 407, 1538, 502, 1760, 576, 307, 556, 2056, 43, 816, 1617, 30, 669, 2051, 1447, 1131, 144, 1791, 219, 1916, 357, 1113, 826, 865, 2004, 272, 150, 213, 399, 1537, 1426, 929, 316, 94, 872, 983, 933, 525, 277, 1993, 1625, 717, 1523, 1563, 69, 1471, 1247, 1296, 908, 460, 1619, 1771, 1359, 955, 471, 1930, 197, 466, 1899, 1898, 544, 1631, 1389, 1388, 1866, 68, 1332, 48, 1612, 2017, 16, 2001, 1553, 1648, 1091, 111, 2058, 285, 664, 134, 1234, 957, 1997, 855, 837, 1053, 1510, 1264, 1851, 609, 2027, 446, 7, 386, 1097, 1362, 477, 211, 166, 611, 836, 1345, 1953, 951, 1767, 1072, 1392, 452, 333, 1151, 1743, 759, 379, 1396, 1999, 1758, 668, 2049, 890, 491, 1189, 516, 1217, 764, 1865, 1502, 2053, 1160, 493, 1232, 537, 1104, 559, 1677, 747, 1041, 1688, 1100, 1304, 199, 679, 832, 2042, 1896, 1134, 795, 661, 829, 638, 1737, 871, 521, 1749, 585, 988, 1109, 961, 1513, 353, 1921, 1964]
print(obn,observenode)

#感染过程
node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
#node1 = list(set(node))#节点元素从小到大排序
#print(len(node))
S = node
I = []

j=0
while j<1:
    #start_node = random.choice(node)#1个初始感染节点
    start_node = 18#1个初始感染节点
    I.append(start_node)
    S.remove(start_node)
    j=j+1
print("start_node:")
print(start_node)
new_G = nx.DiGraph()    ###感染图
sub_new_G_b = nx.DiGraph()   ##观测到的感染图
sub_new_G_w = nx.DiGraph()   ##观测到的感染图
sub_new_G_m = nx.DiGraph() 
count = [1]
statechange = []
edgechange = []
edgeweight = []
weight_s = 1
#for i in range(Roundtime):
while len(I)<=part*len(G.nodes()):
    for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
        if int(nbr) in I:            
            for key in datadict:
                if int(key) in S:
                    rate = G.get_edge_data(int(nbr),int(key))['weight']   #I->S 
                    if random.random() <= rate: 
                        statechange.append(int(key))
                        new_G.add_edge(int(nbr),int(key))     #I->S
    statechange=list(set(statechange))
    for i in statechange:
        S.remove(i)
        I.append(i)
    if len(I)==1:
        break
    count.append(len(I))
    statechange = []

pos = nx.spring_layout(new_G) 
nx.draw(new_G,pos,with_labels = True)  #画图
plt.show()       
print("len(new_G.edges()):",len(new_G.edges()))
ob_I =[]
for gan in observenode:
    if gan in I:
        ob_I.append(gan)
subgraph = G.subgraph(ob_I)    #观测到的I
pos = nx.spring_layout(subgraph) 
nx.draw(subgraph,pos,with_labels = True)  #画图
plt.show()        
print("len(subgraph.edges()):",len(subgraph.edges()))
for u,v,w in subgraph.edges(data=True):
    #print(w)
    sub_new_G_b.add_edge(u,v,weight = round(w['weight'],2))
    sub_new_G_m.add_edge(u,v,weight = 1)
    sub_new_G_w.add_edge(u,v,weight = round(1-w['weight'],2))
                #elif int(key) not in observenode:
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

# #nx.draw(new_G)
# pos = nx.spring_layout(sub_new_G_m) 
# nx.draw(sub_new_G_m,pos,with_labels = True)  #画图
# plt.show()

def jordancenter(G):
    lengths = nx.all_pairs_dijkstra_path_length(G,weight='weight')
    lengths = dict(lengths)
    ec = {}
    #for ei in range(len(lengths)):
    for ei in lengths:       #当观测图的编号不从0按顺序开始时；
        ec[ei]=max(lengths[ei].values())
    radius = min(ec.values())
    p = [v for v in ec if ec[v] == radius]
    return p

# center_frozen_graph = nx.freeze(sub_new_G_m)#G被冷冻为frozen_graph，不会改变
# center_unfrozen_graph = nx.Graph(center_frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
#Gc_node = max(nx.connected_components(sub_new_G_m), key=len)   #sub_newG的最大连通子图来求Jordan Center
#print(Gc_node)
# Gc = center_unfrozen_graph.subgraph(Gc_node)
# Jordan_center  = nx.center(Gc)
jc_b_all = []
for c in nx.weakly_connected_components(sub_new_G_b): #所有联通子图  强联通子图会有很多单个节点
    subgraph = sub_new_G_b.subgraph(c)
    # pos = nx.spring_layout(subgraph) 
    # nx.draw(subgraph,pos,with_labels = True)  #画图
    # plt.show()    
    x= jordancenter(subgraph)
    print("x:",x)
    jc_b_all.extend(x)
print("jc_b_all:",jc_b_all)

jc_m_all = []
for c in nx.weakly_connected_components(sub_new_G_m): #所有联通子图
    subgraph = sub_new_G_m.subgraph(c)
    x= jordancenter(subgraph)
    print("x:",x)
    jc_m_all.extend(x)
print("jc_m_all:",jc_m_all)

jc_w_all = []
for c in nx.weakly_connected_components(sub_new_G_w): #所有联通子图
    subgraph = sub_new_G_w.subgraph(c)
    x= jordancenter(subgraph)
    print("x:",x)
    jc_w_all.extend(x)
print("jc_w_all:",jc_w_all)



# jc_b = jordancenter(sub_new_G_b)
# jc_w = jordancenter(sub_new_G_w)
# jc_m = jordancenter(sub_new_G_m)
# print("jc_b:",jc_b)  #有可能是非完全联通的
# print("jc_w:",jc_w) 
# print("jc_m:",jc_m) 
read_dic = np.load(fname+"_short_path.npy",allow_pickle = True).item()
sumb=0
for i in range(len(jc_b_all)):
    a = read_dic[start_node][jc_b_all[i]]
    sumb = sumb+a
dis_b = sumb/len(jc_b_all)
sumw=0
for i in range(len(jc_w_all)):
    a = read_dic[start_node][jc_w_all[i]]
    sumw = sumw+a
dis_w = sumw/len(jc_w_all)
summ=0
for i in range(len(jc_m_all)):
    a = read_dic[start_node][jc_m_all[i]]
    summ = summ+a
dis_m = summ/len(jc_m_all)

print("dis_b:",dis_b)
print("dis_m:",dis_m)
print("dis_w:",dis_w)

jc_c_all = []
for c in nx.strongly_connected_components(sub_new_G_m): #所有联通子图
    subgraph = sub_new_G_m.subgraph(c)
    pos = nx.spring_layout(subgraph) 
    nx.draw(subgraph,pos,with_labels = True)  #画图
    plt.show()    
    x= nx.center(subgraph)      #该方法只有对于strongly联通子图才能够使用，所以191行为weakly时bug，为strongly才行
    print("x:",x)
    jc_c_all.extend(x)
print("jc_c_all:",jc_c_all)
print("jc_c_all_dis:",read_dic[start_node][jc_c_all[0]])
#jc_sub_m = nx.center(sub_new_G_m)     #有可能是非完全连接图
#print("jc_sub_m:",jc_sub_m)
#print("jc_sub_m_dis:",read_dic[start_node][jc_sub_m[0]])


# jc_inf = nx.center(new_G)
# print("jc_inf:",jc_inf)
# print("jc_inf_dis:",read_dic[start_node][jc_inf[0]])

#在一张图上显示
Gt=new_G.copy()
for node in I:
    if node!= start_node:
        Gt.remove_node(node)
#print(len(Gt.nodes()))#源节点
#print(len(G.nodes()))
pos = nx.spring_layout(G)
nx.draw(G,pos,node_color='b',node_size=1,edge_color = 'b',with_labels=True)
nx.draw_networkx_nodes(new_G,pos,node_color='r',node_size=1)
nx.draw_networkx_edges(new_G,pos,edge_color = 'r')
nx.draw_networkx_nodes(sub_new_G_b,pos,node_color='y',node_size=1)
nx.draw_networkx_edges(sub_new_G_b,pos,edge_color = 'y')
nx.draw(Gt,pos,node_color = 'g',node_size = 10)
plt.show()