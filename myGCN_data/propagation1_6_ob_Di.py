#有向图
#给感染节点一个label， 感染状态作为node_labels
#部分观测
#新的感染图画法，添加边根据所有感染节点来增加
#propagation_pro1.py的升级
#给边增加权重，随机增加
#改为传播到总结点数的0.8停止传播
#propagation使用的是从I开始遍历，使S改变状态
#本程序从S开始遍历，感染概率为1-（1-q）^n
#在一张图上同时显示初始图结构和传播感染图，即传播感染图是初始图结构的一部分
#from research.new_propagation_SI import InfectionRate, Roundtime
from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx
import os
import random
from itertools import islice
import argparse
from scipy.sparse.coo import coo_matrix
from sklearn.semi_supervised import LabelSpreading

N =2068
name = "p2p2068"
B = np.load(name+".npy")#读取固定的图
Gnw = nx.DiGraph(B)
# adj_matrix = nx.adjacency_matrix(G).todense()
# coo_A=coo_matrix(adj_matrix)   #邻接矩阵的边的行/列的坐标
# edge_index = [coo_A.row+1,coo_A.col+1]
# print(len(edge_index))
G = nx.read_edgelist('./'+name+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())
print(len(G.edges()))
# pos = nx.spring_layout(G) #为什么报错！！！！！！！！！！
# nx.draw(G,pos,with_labels = True)#逗号是中文的  #画图
# plt.show()
obn = int(N*0.5)  #有一半的节点可以观测到
observenode = random.sample(range(0,N-1),obn)#每次都随机选还是固定呢
#observenode= [i for i in range(62)]
#observenode=[21, 38, 27, 42, 55, 16, 58, 60, 28, 49, 26, 51, 45, 17, 11, 33, 41, 57, 39, 10, 18, 34, 25, 36, 20, 53, 56, 23, 3, 24, 52]#62,0.5
#observenode = [13, 27, 62, 43, 23, 0, 36, 38, 63, 11, 2, 8, 65, 59, 16, 10, 48, 55, 28, 45, 52, 32, 41, 25, 42, 24, 22, 19, 50, 53, 46, 33, 18]#67,0.5
#2068,ob0.5
#observenode  =[903, 1578, 1328, 83, 2006, 1269, 1256, 19, 1288, 73, 762, 182, 523, 444, 174, 875, 884, 1445, 1596, 1443, 1304, 1334, 1254, 1625, 798, 93, 899, 694, 1980, 1464, 882, 944, 1044, 413, 1723, 451, 1713, 621, 1028, 1983, 1531, 1700, 622, 1673, 755, 1077, 963, 673, 1595, 1159, 1210, 352, 446, 377, 627, 878, 1883, 1672, 4, 342, 770, 1683, 388, 1584, 618, 254, 1415, 1486, 1425, 221, 52, 1789, 220, 1344, 847, 216, 1253, 2033, 1456, 344, 769, 677, 1697, 1185, 1861, 496, 1255, 684, 1005, 1149, 1632, 1911, 1610, 1407, 362, 1621, 809, 103, 82, 1119, 303, 1838, 1644, 394, 1812, 1642, 1199, 1998, 867, 949, 685, 1126, 1658, 640, 1348, 717, 1966, 804, 80, 550, 224, 1260, 1813, 822, 1692, 1541, 79, 1956, 422, 806, 21, 461, 475, 1667, 866, 612, 2065, 558, 134, 159, 716, 1660, 1096, 574, 1761, 346, 466, 106, 1745, 1579, 576, 1042, 1987, 597, 1727, 498, 945, 489, 1087, 1440, 122, 158, 1552, 1638, 675, 1515, 569, 202, 1388, 1898, 380, 1384, 1882, 445, 1138, 1755, 233, 588, 335, 1576, 834, 1743, 881, 1853, 1721, 1843, 2024, 529, 1637, 143, 1362, 328, 416, 1252, 1430, 1136, 858, 1703, 1002, 1069, 811, 1078, 100, 705, 1917, 1992, 31, 727, 1680, 1528, 1076, 1532, 1325, 157, 1354, 296, 1868, 773, 633, 1742, 1655, 827, 1305, 1546, 2045, 217, 647, 1754, 736, 2060, 468, 1487, 288, 1019, 605, 429, 1824, 1952, 40, 48, 982, 524, 1036, 954, 740, 1895, 1394, 6, 1227, 1030, 1327, 185, 1079, 919, 211, 836, 1009, 1708, 1849, 1516, 1208, 126, 1314, 844, 1197, 2034, 1007, 105, 1591, 2057, 76, 1306, 2013, 738, 456, 742, 160, 404, 1791, 786, 1442, 396, 244, 213, 1871, 1969, 449, 154, 735, 1690, 228, 437, 1948, 1864, 859, 269, 1004, 848, 1158, 2005, 152, 1781, 1175, 639, 364, 136, 928, 1846, 1827, 729, 595, 669, 1819, 1504, 1439, 1109, 948, 1221, 937, 1273, 1828, 1385, 146, 306, 1113, 153, 791, 1495, 442, 838, 1793, 1982, 1641, 1409, 248, 176, 749, 324, 28, 472, 1228, 2016, 1315, 869, 321, 511, 249, 88, 343, 17, 1100, 810, 2041, 1472, 247, 1548, 1527, 2002, 503, 195, 1071, 1144, 1226, 393, 929, 407, 54, 658, 417, 1841, 1677, 1712, 1905, 772, 1669, 408, 600, 222, 1434, 187, 1095, 325, 779, 1041, 318, 1088, 1248, 709, 512, 1264, 1474, 10, 894, 1670, 1570, 502, 236, 260, 499, 1436, 703, 1507, 1744, 1365, 589, 1166, 829, 1271, 1192, 1934, 824, 931, 386, 290, 797, 167, 984, 978, 71, 1053, 1639, 581, 1777, 1577, 1141, 681, 87, 849, 746, 177, 1331, 759, 801, 1820, 42, 1741, 534, 1220, 1751, 3, 520, 309, 170, 1022, 1082, 671, 1859, 1484, 1417, 1602, 1435, 1674, 961, 767, 1706, 1542, 696, 425, 1222, 1432, 1575, 229, 987, 1459, 1350, 974, 2063, 1899, 850, 1876, 1910, 1648, 1418, 1624, 235, 307, 448, 1090, 55, 460, 946, 666, 653, 1215, 1198, 880, 2004, 339, 651, 1043, 463, 1985, 1335, 256, 1381, 64, 1313, 1489, 730, 1891, 1603, 70, 854, 231, 1467, 765, 215, 1061, 1888, 1281, 1250, 1734, 1057, 604, 918, 376, 355, 1798, 2054, 410, 1553, 1569, 1821, 1896, 1500, 699, 896, 1568, 471, 108, 491, 1465, 840, 1961, 554, 485, 1429, 1121, 1901, 874, 1295, 168, 902, 1160, 933, 648, 477, 2046, 1452, 1186, 1239, 1779, 1203, 356, 301, 634, 1650, 1839, 656, 1211, 1485, 1510, 1678, 1157, 819, 888, 1759, 775, 330, 877, 1237, 761, 766, 555, 665, 1549, 1614, 495, 2025, 1482, 2037, 390, 1064, 279, 58, 906, 1073, 1116, 1646, 2031, 515, 1715, 112, 398, 830, 1062, 936, 116, 418, 952, 825, 1054, 1499, 764, 1738, 1693, 725, 938, 1765, 67, 115, 166, 291, 20, 401, 1103, 643, 1520, 2064, 1696, 870, 1218, 9, 500, 592, 1815, 1297, 226, 958, 434, 1768, 1562, 1872, 1902, 26, 652, 400, 141, 1530, 150, 712, 803, 1386, 1368, 186, 1881, 199, 348, 1235, 1762, 584, 57, 1371, 259, 1597, 139, 691, 873, 1155, 1647, 1056, 700, 1829, 1514, 1689, 341, 1114, 102, 885, 1867, 1193, 316, 753, 1080, 851, 16, 782, 272, 1953, 1282, 1914, 1172, 720, 1946, 1251, 218, 1200, 1943, 942, 1811, 300, 883, 547, 1267, 1473, 1183, 1323, 1756, 1018, 1845, 624, 1125, 841, 1243, 1130, 1746, 1814, 890, 156, 907, 1571, 955, 860, 1420, 2062, 1346, 467, 431, 1449, 294, 1559, 1758, 283, 1840, 1662, 1686, 113, 292, 1990, 1177, 1181, 433, 959, 1932, 2028, 1343, 610, 676, 852, 488, 1403, 1920, 833, 546, 1234, 1112, 1137, 1608, 514, 1547, 172, 1191, 596, 501, 1535, 557, 409, 171, 371, 419, 1850, 454, 1724, 957, 1664, 97, 35, 1055, 1676, 207, 484, 988, 579, 1620, 835, 1340, 1396, 1101, 678, 654, 1023, 436, 63, 1545, 1246, 1184, 1612, 752, 1291, 751, 189, 1601, 1224, 1809, 932, 1247, 459, 580, 788, 1330, 807, 1709, 1951, 1083, 1176, 1942, 1047, 209, 1923, 910, 1659, 1400, 1147, 1475, 1111, 173, 2014, 525, 1940, 2059, 704, 2020, 1447, 38, 1025, 533, 161, 549, 1780, 1469, 1605, 242, 1629, 1833, 1893, 1705, 544, 1564, 689, 1258, 1908, 1263, 266, 737, 1311, 839, 1592, 826, 1132, 960, 1318, 917, 629, 613, 1617, 1933, 1707, 1195, 1786, 121, 2058, 1212, 552, 43, 950, 1508, 223, 239, 1257, 1379, 1719, 1375, 1145, 1317, 1364, 1736, 1015, 1679, 1497, 1618, 1268, 1897, 1127, 1333, 743, 994, 1551, 537, 1294, 81, 243, 1630, 1583, 1737, 626, 646, 598, 661, 389, 1941, 1461, 1206, 297, 131, 326, 332, 893, 1800, 1947, 1518, 197, 2043, 1352, 2047, 1024, 895, 1342, 1194, 714, 1133, 1566, 1636, 780, 1122, 939, 2010, 1805, 51, 711, 1476, 659, 1984, 756, 1060, 744, 333, 638, 1478, 1631, 1929, 1773, 1816, 263, 924, 2053, 320, 1032, 397, 1353, 1860, 1048, 95, 179, 353, 1710, 1968, 1623, 1626, 531, 1927, 383, 941, 246, 78, 1017, 832, 178, 1367, 817, 1863, 1397, 366, 1099, 414, 1848, 1997, 1807, 962, 1201, 1852, 1916, 1046, 1033, 476, 565, 395, 261, 571, 2051, 360, 637, 110, 734, 1975, 1808, 816, 993, 1795, 1016, 1242, 556, 1173, 1390, 1817, 741, 98, 225, 450, 1903, 758, 857, 802, 453, 133, 1171, 1470, 368, 86, 855, 1775, 1695, 528, 1606, 2017, 812, 1554, 145, 469, 504, 50, 1725, 799, 868, 84]
print(obn,observenode)

InfectionRate = 0.1#概率太大，10轮感染1400个节点
Roundtime =3
part = 0.1

#感染过程
node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
#node1 = list(set(node))#节点元素从小到大排序
#print(len(node))
S = node
I = []

j=0
while j<1:
    #start_node = random.choice(node)#1个初始感染节点
    start_node =25#1个初始感染节点
    I.append(start_node)
    S.remove(start_node)
    j=j+1
print("start_node:")
print(start_node)

new_G_A = nx.DiGraph()
ob_G = nx.DiGraph()
count = [1]
statechange = []
edgechange = []
edgeweight = []
weight_s = 1
for r in range(Roundtime):       #####从I开始遍历，小于边的概率就感染
#while len(I)<=part*len(G.nodes()):
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
# #for i in range(Roundtime):     #从S开始遍历，有向时矛盾
# while len(I)<=part*len(G.nodes()):
#     for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
#         if int(nbr) in S:
#             node_adj = 0               #S节点的感染邻接点数
#             for key in datadict:
#                 if int(key) in I:
#                     node_adj=node_adj+1
#                     edgechange.append(int(key))
#                     edgeweight.append(G.get_edge_data(int(key),int(nbr))['weight'])     #I感染S节点，key->nbr
#             for weight in edgeweight:
#                 weight_s = weight_s*(1-weight)
#             rate = 1-weight_s
#             if random.random() <= rate:   #被感染后，节点状态变化，感染图的边增加（周围所有感染节点与该点的连边都算上）
#                 for a in edgechange:
#                     new_G.add_edge(a,int(nbr))     #key->nbr==I->S
#                 statechange.append(int(nbr))
#             edgechange = []
#             edgeweight=[]
#             weight_s=1      
#     for i in statechange:
#         S.remove(i)
#         I.append(i)

#     if len(I)==1:
#         break
#     count.append(len(I))
#     statechange = []
pos = nx.spring_layout(new_G_A) #为什么报错！！！！！！！！！！
nx.draw(new_G_A,pos,with_labels = True)#逗号是中文的  #画图
plt.show()
print("len(new_G_A.edges()):",len(new_G_A.edges()))
#连接所有I的节点的边

subgraph = G.subgraph(I)
pos = nx.spring_layout(subgraph) #为什么报错！！！！！！！！！！
nx.draw(subgraph,pos,with_labels = True)#逗号是中文的  #画图
plt.show()

pos = nx.spring_layout(ob_G) #为什么报错！！！！！！！！！！
nx.draw(ob_G,pos,with_labels = True)#逗号是中文的  #画图
plt.show()


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
print('len(new_G_A.nodes()):')
print(len(new_G_A.nodes()))
# print('edges:')
# print(new_G.edges())

# #nx.draw(new_G)
# pos = nx.spring_layout(new_G) #为什么报错！！！！！！！！！！
# nx.draw(new_G,pos,with_labels = True)#逗号是中文的  #画图
# plt.show()

# #在一张图上显示
# Gt=new_G.copy()
# for node in I:
#     if node!= start_node:
#         Gt.remove_node(node)
# #print(len(Gt.nodes()))#源节点
# #print(len(G.nodes()))
# pos = nx.spring_layout(G)
# print("len(G.edges():",len(G.edges()))
# nx.draw(G,pos,node_color='b',node_size=1,edge_color = 'b',with_labels=True)
# #nx.draw_networkx_nodes(observenode,pos,node_color='r',node_size=5)
# nx.draw_networkx_nodes(I,pos,node_color='g',node_size=15)
# nx.draw_networkx_edges(subgraph,pos,edge_color = 'r')
# nx.draw(Gt,pos,node_color = 'red',node_size = 10,with_labels=True)
# plt.show()
